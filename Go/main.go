package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func makeRange(min, max, step int) []int {
	a := []int{}

	for i := 0; i < max; i += step {
		a = append(a, min+i)
	}
	return a

}

type Dataset struct {
	data            *mat.Dense
	entropy         float64
	targetAttribute string   // car_eval
	attributes      []string // nazivi atributa za X
	targetValues    []int    // vrijednosti 0-3 za car_eval
}

func (dataset *Dataset) doBagging(samplesNum, featuresNum int) *Dataset {
	samplesTotal := dataset.data.ColView(0).Len()
	//attributesLen := dataset.data.RowView(0).Len()

	attrs := []int{0, 1, 2, 3, 4, 5}
	attributesIndices := []int{}
	newAttributes := []string{}

	rand.Shuffle(len(attrs), func(i, j int) { attrs[i], attrs[j] = attrs[j], attrs[i] })
	attributesIndices = attrs[:featuresNum]

	newMatrix := mat.NewDense(samplesNum, featuresNum+1, nil)

	for ind, attrIndex := range attributesIndices {
		newAttributes = append(newAttributes, dataset.attributes[ind])

		for i := 0; i < samplesNum; i++ {
			sampleIndex := rand.Intn(samplesTotal - 1)
			newMatrix.Set(i, ind, dataset.data.At(sampleIndex, attrIndex))
		}
	}

	return &Dataset{data: newMatrix, entropy: 0, targetAttribute: dataset.targetAttribute, attributes: newAttributes, targetValues: dataset.targetValues}
}

func (dataset *Dataset) getPureLabel() float64 {
	return dataset.data.ColView(dataset.data.RowView(0).Len() - 1).AtVec(0)
}

func (dataset *Dataset) isPure() bool {
	first_label := dataset.getPureLabel()

	for i := 0; i < dataset.data.ColView(dataset.data.RowView(0).Len()-1).Len(); i++ {
		if first_label != dataset.data.ColView(dataset.data.RowView(0).Len()-1).AtVec(i) {
			return false
		}
	}
	return true
}

// metoda za uklanjanje vrijednosti iz vektora

func removeElement(vec mat.Vector, index int) *mat.VecDense {

	newVector := mat.NewVecDense(vec.Len()-1, nil)

	for i := 0; i < vec.Len(); i++ {
		if i > index {
			newVector.SetVec(i-1, vec.AtVec(i))
		} else if i < index {
			newVector.SetVec(i, vec.AtVec(i))
		}
	}

	return newVector
}

func (dataset *Dataset) split(splitOnAttribute string) map[int]*Dataset {
	splittedDatasets := make(map[int][]*mat.VecDense)
	rtnVal := make(map[int]*Dataset)
	attributeIndex := 0

	newAttributes := []string{}

	for ind, val := range dataset.attributes {
		if val == splitOnAttribute {
			attributeIndex = ind
		} else {
			newAttributes = append(newAttributes, val)
		}
	}

	// vracamo mapu sa value : Dataset
	for i := 0; i < dataset.data.RowView(attributeIndex).Len(); i++ {
		val := int(dataset.data.RowView(attributeIndex).AtVec(i))

		if _, ok := splittedDatasets[val]; ok {
			splittedDatasets[val] = append(splittedDatasets[val], removeElement(dataset.data.ColView(i), attributeIndex))
		} else {
			splittedDatasets[val] = []*mat.VecDense{}
			splittedDatasets[val] = append(splittedDatasets[val], removeElement(dataset.data.ColView(i), attributeIndex))
		}
	}

	// prodjes kroz mapu i napravis nove datasetove
	for key, value := range splittedDatasets {
		newMatrix := mat.NewDense(len(value), value[0].Len(), nil)
		for i := 0; i < len(value); i++ {
			for j := 0; j < value[0].Len(); j++ {
				newMatrix.Set(i, j, value[i].AtVec(j))
			}
		}

		newDataset := Dataset{newMatrix, float64(0), dataset.targetAttribute, newAttributes, dataset.targetValues}
		rtnVal[key] = &newDataset
	}

	return rtnVal
}

func (dataset *Dataset) calcEntropy() float64 {

	target_col := dataset.data.ColView(dataset.data.RowView(0).Len() - 1)
	total_num := float64(target_col.Len())

	p_values := []float64{}

	for i := 0; i < len(dataset.targetValues); i++ {
		value := dataset.targetValues[i]
		var counter float64 = 0
		for j := float64(0); j < total_num; j++ {
			if int(dataset.data.ColView(dataset.data.RowView(0).Len()-1).AtVec(i)) == value {
				counter += 1
			}
		}
		p_values = append(p_values, counter/total_num)
	}

	dataset.entropy = stat.Entropy(p_values) // prima []float64

	return dataset.entropy
}

func (dataset *Dataset) calcGainRatio(splitOnAttribute string) (float64, map[int]*Dataset) {
	entropyBefore := dataset.calcEntropy()
	splittedMap := dataset.split(splitOnAttribute)

	samplesTotalNo, _ := dataset.data.Dims()
	samplesTotalNum := float64(samplesTotalNo)

	entropyAfter := float64(0)
	splitInfo := float64(0.0000000001)

	for _, val := range splittedMap {
		entropyAfter += val.calcEntropy()
		samplesNo, _ := val.data.Dims()
		samplesNum := float64(samplesNo)
		splitInfo += (samplesNum / samplesTotalNum * math.Log2(samplesNum/samplesTotalNum))
	}

	gainRatio := (entropyBefore - entropyAfter) / splitInfo
	return gainRatio, splittedMap
}

type Node struct {
	dataset           *Dataset
	children          []*Node
	decisionAttribute string
	value             int
	label             int
}

func (node *Node) fit() {
	if node.dataset.isPure() || len(node.dataset.attributes) == 0 {
		node.label = int(node.dataset.getPureLabel())
		return
	}

	attributeScores := make(map[string]map[int]*Dataset)
	maxGainRatio := math.Inf(-1)
	maxAttribute := ""

	for _, attr := range node.dataset.attributes {
		attrGainRatio, splittedDict := node.dataset.calcGainRatio(attr)
		attributeScores[attr] = splittedDict

		if maxGainRatio < attrGainRatio {
			maxGainRatio = attrGainRatio
			maxAttribute = attr
		}
	}

	node.decisionAttribute = maxAttribute

	for key, val := range attributeScores[maxAttribute] {
		child := Node{dataset: val, value: key}
		node.children = append(node.children, &child)
		child.fit()
	}
}

func (node *Node) isLeaf() bool {
	return len(node.children) == 0
}

func (node *Node) predict(example *mat.VecDense) int {
	if node.isLeaf() == true {
		return node.label
	}

	attributeIndex := getAttributeIndex(node.decisionAttribute)

	for _, node := range node.children {
		if example.AtVec(attributeIndex) == float64(node.value) {
			return node.predict(example)
		}
	}

	return -13
}

func getAttributeIndex(attribute string) int {
	attrList := []string{"buying", "maint", "doors", "persons", "lug_boot", "safety"}

	for ind, val := range attrList {
		if val == attribute {
			return ind
		}
	}

	return -1

}

type Tree struct {
	rootNode *Node
	dataset  *Dataset
}

func (tree *Tree) fit() {
	tree.rootNode = &Node{dataset: tree.dataset, decisionAttribute: ""}
	tree.rootNode.fit()
}

func (tree *Tree) predict(example *mat.VecDense) int {
	return tree.rootNode.predict(example)
}

type RandomForest struct {
	treesNum int
	trees    []*Tree
	dataset  *Dataset
}

func (forest *RandomForest) fit() {
	for i := 0; i < forest.treesNum; i++ {
		tree := Tree{dataset: forest.dataset.doBagging(500, 4)}
		forest.trees = append(forest.trees, &tree)
		tree.fit()
	}

}

func (forest *RandomForest) fit_trees_parallel(trees []*Tree, wg *sync.WaitGroup) {

	for _, tree := range trees {
		tree.fit()
	}
	wg.Done()

}

var wg sync.WaitGroup

func (forest *RandomForest) fitParallel(numOfProcesses int) {

	for i := 0; i < forest.treesNum; i++ {
		tree := Tree{dataset: forest.dataset.doBagging(500, 4)}
		forest.trees = append(forest.trees, &tree)

	}

	treesPerTask := len(forest.trees) / numOfProcesses

	for i := 0; i < numOfProcesses; i++ {
		wg.Add(1)
		go forest.fit_trees_parallel(forest.trees[i*treesPerTask:(1+i)*treesPerTask], &wg)
	}
	wg.Wait()
}

func (forest *RandomForest) predict(example *mat.VecDense) int {
	results := make(map[int]float64)

	for _, val := range forest.dataset.targetValues {
		results[val] = float64(0)
	}

	for _, tree := range forest.trees {
		inference := tree.predict(example)

		if inference != -13 {
			results[inference] += 1
		}
	}

	maxScore := math.Inf(-1)
	maxAttr := 0

	for key, val := range results {
		if val >= maxScore {
			maxScore = val
			maxAttr = key
		}
	}

	return maxAttr

}

func weakScaling(numOfTrees, numOfProcesses []int) map[string]float64 {

	dataset := loadData()
	results := make(map[string]float64)

	for ind, tasks := range numOfProcesses {
		startTime := time.Now()
		forest := RandomForest{treesNum: numOfTrees[ind], dataset: dataset}
		forest.fitParallel(tasks)
		key := "(" + strconv.Itoa(tasks) + "," + strconv.Itoa(numOfTrees[ind]) + ")"
		results[key] = toFixed(time.Since(startTime).Seconds(), 5)
	}

	return results

}

func strongScaling(numOfProcesses []int) map[string]float64 {

	dataset := loadData()
	results := make(map[string]float64)

	for _, tasks := range numOfProcesses {
		startTime := time.Now()
		forest := RandomForest{treesNum: 700, dataset: dataset}
		forest.fitParallel(tasks)
		results[strconv.Itoa(tasks)] = toFixed(time.Since(startTime).Seconds(), 5)
	}

	return results
}

func writeResultsToFile(results map[string]float64, scalingType string) {

	jsonString, err := json.Marshal(results)
	check(err)

	file, err := os.OpenFile("./data/"+scalingType+".txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	check(err)

	file.Write(jsonString)

	file.Close()

}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func generateTreeDescription(nodes []*Node, output string) string {

	for ind, node := range nodes {
		if ind == len(nodes)-1 {
			output += node.decisionAttribute + " " + strconv.Itoa(node.value) + " " + strconv.Itoa(node.label) + ";"
		} else {
			output += node.decisionAttribute + " " + strconv.Itoa(node.value) + " " + strconv.Itoa(node.label) + ","
		}
	}

}

func writeTreesToFile(trees []*Tree) string {

	output := ""

	for _, tree := range trees {
		output += tree.rootNode.decisionAttribute + " " + strconv.Itoa(tree.rootNode.value) + " " + strconv.Itoa(tree.rootNode.label) + ";"

	}

	return output

}

func loadData() *Dataset {
	csvfile, err := os.Open("data/car_encoded.csv")
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(csvfile)

	dataMatrix := mat.NewDense(df.Nrow(), df.Ncol(), nil)

	for i := 0; i < df.Nrow(); i++ {

		for j := 0; j < df.Ncol(); j++ {
			dataMatrix.Set(i, j, df.Elem(i, j).Float())
		}
	}

	attributes := df.Names()[:6]
	datasetStart := Dataset{data: dataMatrix, entropy: float64(0), targetAttribute: "car_eval", attributes: attributes, targetValues: []int{0, 1, 2, 3}}

	return &datasetStart
}

// Preuzeto iz projektnog primjera
func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}

// Preuzeto iz projektnog primjera
func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

func main() {

	numOfProcesses := makeRange(1, 16, 1)

	numOfTrees := makeRange(100, 3200, 200)
	resultsWeak := weakScaling(numOfTrees, numOfProcesses)
	writeResultsToFile(resultsWeak, "weakScaling")

	resultsStrong := strongScaling(numOfProcesses)
	writeResultsToFile(resultsStrong, "strongScaling")
	fmt.Println("Results written...")
}
