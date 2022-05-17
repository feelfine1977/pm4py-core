%Nodes: 12 Edges: 30
node(0).
node(1).
node(2).
node(3).
node(4).
node(5).
node(6).
node(7).
node(8).
node(9).
node(10).
node(11).
labelname(1,"DRIVE").
labelname(2,"LOAD").
labelname(3,"UNLOAD").
%% Left: P1 C2. 
edge((0,1)).
tlabel((0,1),1).
edge((1,0)).
tlabel((1,0),1).

edge((0,2)).
tlabel((0,2),1).
edge((2,0)).
tlabel((2,0),1).

edge((2,1)).
tlabel((2,1),1).
edge((1,2)).
tlabel((1,2),1).

%% Center: P1 at truck.
edge((3,4)).
tlabel((3,4),1).
edge((4,3)).
tlabel((4,3),1).

edge((3,5)).
tlabel((3,5),1).
edge((5,3)).
tlabel((5,3),1).

edge((4,5)).
tlabel((4,5),1).
edge((5,4)).
tlabel((5,4),1).

%% Right: at P1 c1.
edge((9,11)).
tlabel((9,11),1).
edge((11,9)).
tlabel((11,9),1).

edge((9,10)).
tlabel((9,10),1).
edge((10,9)).
tlabel((10,9),1).

edge((10,11)).
tlabel((10,11),1).
edge((11,10)).
tlabel((11,10),1).

%% Bottom: at p1 c3
edge((6,7)).
tlabel((6,7),1).
edge((7,6)).
tlabel((7,6),1).

edge((6,8)).
tlabel((6,8),1).
edge((8,6)).
tlabel((8,6),1).

edge((7,8)).
tlabel((7,8),1).
edge((8,7)).
tlabel((8,7),1).


%% From the center ( unload / load )
% Unload C2
edge((3,1)).
tlabel((3,1),3). 
% Load
edge((1,3)).
tlabel((1,3),2). 


% Unload C1
edge((4,9)).
tlabel((4,9),3). 
% Load
edge((9,4)).
tlabel((9,4),2). 


% Unload C3
edge((5,8)).
tlabel((5,8),3). 
% Load
edge((8,5)).
tlabel((8,5),2). 
