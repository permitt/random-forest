
tree := 'C:\Users\Ritam\Projects\Machine Learning\Masinsko Ucenje\machine-learning\random_forest\data\forest.json' asFileReference
	readStreamDo: [ :readStream |
		STONJSON fromStream: readStream ] .
	


root_node := tree at: 'root-0'.
child := root_node at: 'decision_attribute'.
children:=root_node at: 'children'.

shape := RTCompositeShape new.
shape add: (RTLabel new color: Color gray; text: [:aValue | 'Attr: ', aValue asString ]).
shape add: (RTBox new
				color: Color lightRed;
				width: 20;
				height: 20).

v := RTView new.
shape_line := RTArrowedLine  new color: Color gray.

nodes := Set new.

cnt :=0.
cnt1 := 0.
cnt2 := 0.

tree do: [ :root_node | 
	
		Transcript show: 'ROOT: ', (root_node at: 'decision_attribute');cr.
		
		elementShape := RTBox new size: 20; color: Color blue trans.
		e1 := elementShape elementOn: 'Root-attr: ', (root_node at: 'decision_attribute').
		v add: e1.
		e1 @ RTLabeled.
		e1 @ RTDraggable.
		e1 translateTo: 650 * cnt @ 0.
		
		cnt := cnt + 1.

		
		(root_node at: 'children') do: [
				:child_node | 
				elementShape := RTBox new size: 20; color: Color blue trans.
				Transcript show: 'CHILD: ', (child_node at: 'decision_attribute');cr.
				e2 := elementShape elementOn: 'Attr: ', (child_node at: 'decision_attribute'), ' Val: ', (child_node at: 'value').
				v add: e2.
				e2 @ RTLabeled.
				e2 @ RTDraggable.
				e2 translateTo: -60 + (140 * cnt1) @ 80.
		
				cnt1 := cnt1 + 1.
				v add: (shape_line edgeFrom: e1 to: e2).
		
				
				
				
				(child_node at: 'children') do: [ 
						:grandkid_node | 
						
						Transcript show: 'GRANDKID: ',(grandkid_node at: 'value');cr.

				elementShape := RTBox new size: 20; color: Color green trans.
				e3 := elementShape elementOn: 'Attr: ', (grandkid_node at: 'decision_attribute'), ' Val: ', (grandkid_node at: 'value').
				v add: e3.
				e3 @ RTLabeled.
				e3 @ RTDraggable.
				e3 translateTo: -250 + (100 * cnt2) @ 150.
				v add: (shape_line edgeFrom: e2 to: e3).
				cnt2:=cnt2+1.
				
				]
			]
		].
	
	v				
			
		
	

