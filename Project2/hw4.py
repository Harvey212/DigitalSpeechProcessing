import numpy as np
###########################################################
#transition
TransPTo1From=dict()
TransPTo2From=dict()
TransPTo3From=dict()
##############################
TransPTo1From['1'] =0.6
TransPTo1From['2'] =0.5
TransPTo1From['3'] =0.4

TransPTo2From['1'] =0.2
TransPTo2From['2'] =0.3
TransPTo2From['3'] =0.1

TransPTo3From['1'] =0.2
TransPTo3From['2'] =0.2
TransPTo3From['3'] =0.5
#T =   np.array([[0.6, 0.2, 0.2],[0.5, 0.3, 0.2],[0.4, 0.1, 0.5]])
########################################################
#UP DOWN UNCHANGED
EventP1 =dict()
EventP2 =dict()
EventP3 =dict()

EventP1['up'] = 0.7
EventP1['down'] = 0.1
EventP1['unchanged'] = 0.2

EventP2['up'] = 0.1
EventP2['down'] = 0.6
EventP2['unchanged'] = 0.3

EventP3['up'] = 0.3
EventP3['down'] = 0.3
EventP3['unchanged'] = 0.4
######################################################
#observation
######################################################
obseq =['up', 'up', 'unchanged', 'down', 'unchanged', 'down', 'up']
obnum=len(obseq)
########################################################
#initialization
#########################################################
ob=obseq[0]
Probtill1=0.5*EventP1[ob]
Probtill2=0.2*EventP2[ob]
Probtill3=0.3*EventP3[ob]

#to record the largest path prob
LargeProb1=0.5*EventP1[ob]
LargeProb2=0.2*EventP2[ob]
LargeProb3=0.3*EventP3[ob]

#Memory
Memory1=['1']
Memory2=['2']
Memory3=['3']
##########################################################
#################################################################
#for every iteration, every state has:
#1. the largest prob of a path to that state as final state ##prob
#2. total prob to that state ## prob
#3. state sequence of max prob until that state as fin state
##################################################################
for i in range(1, obnum):
	#print("round: {}".format(i))
	newob =obseq[i]
	##############################################################################################################
	#Discuss STATE1
	#PART1: for total prob 
	P11=Probtill1 * TransPTo1From['1']
	P12=Probtill2 * TransPTo1From['2']
	P13=Probtill3 * TransPTo1From['3']

	newProbtill1 = (P11+P12+P13) * EventP1[newob]

	#PART2: for largest prob
	p11=LargeProb1 * TransPTo1From['1']
	p12=LargeProb2 * TransPTo1From['2']
	p13=LargeProb3 * TransPTo1From['3']

	if (p11 > p12) and (p11> p13):
		newMemory1 =Memory1.copy()
		newMemory1.append('1')

		newLargeProb1 = p11* EventP1[newob]
	elif (p12 > p11) and (p12>p13):
		newMemory1 =Memory2.copy()
		newMemory1.append('1')

		newLargeProb1 = p12* EventP1[newob]
	else:
		newMemory1 =Memory3.copy()
		newMemory1.append('1')

		newLargeProb1 = p13* EventP1[newob]

	
	#####################################################
	#Discuss STATE2
	#PART1: for total prob 

	P21=Probtill1 * TransPTo2From['1']
	P22=Probtill2 * TransPTo2From['2']
	P23=Probtill3 * TransPTo2From['3']

	newProbtill2 = (P21+P22+P23) * EventP2[newob]

	#PART2: for largest prob
	p21=LargeProb1 * TransPTo2From['1']
	p22=LargeProb2 * TransPTo2From['2']
	p23=LargeProb3 * TransPTo2From['3']

	if (p21 > p22) and (p21> p23):
		newMemory2 =Memory1.copy()
		newMemory2.append('2')

		newLargeProb2 = p21* EventP2[newob]
	elif (p22 > p21) and (p22>p23):
		newMemory2 =Memory2.copy()
		newMemory2.append('2')

		newLargeProb2 = p22* EventP2[newob]
	else:
		newMemory2 =Memory3.copy()
		newMemory2.append('2')

		newLargeProb2 = p23* EventP2[newob]

	#########################################################
	#Discuss STATE3
	#PART1: for total prob 

	P31=Probtill1 * TransPTo3From['1']
	P32=Probtill2 * TransPTo3From['2']
	P33=Probtill3 * TransPTo3From['3']

	newProbtill3 = (P31+P32+P33) * EventP3[newob]

	#PART2: for largest prob
	p31=LargeProb1 * TransPTo3From['1']
	p32=LargeProb2 * TransPTo3From['2']
	p33=LargeProb3 * TransPTo3From['3']

	if (p31 > p32) and (p31> p33):
		newMemory3 =Memory1.copy()
		newMemory3.append('3')

		newLargeProb3 = p31* EventP3[newob]
	elif (p32 > p31) and (p32>p33):
		newMemory3 =Memory2.copy()
		newMemory3.append('3')

		newLargeProb3 = p32* EventP3[newob]
	else:
		newMemory3 =Memory3.copy()
		newMemory3.append('3')

		newLargeProb3 = p33* EventP3[newob]

	
	############################################################
	#UPDATE TOTAL PROB
	Probtill1 = newProbtill1
	Probtill2 = newProbtill2
	Probtill3 = newProbtill3

	#UPDATE LARGEST PROB
	Memory1 = newMemory1.copy()
	Memory2 = newMemory2.copy()
	Memory3 = newMemory3.copy()

	LargeProb1 = newLargeProb1
	LargeProb2 = newLargeProb2
	LargeProb3 = newLargeProb3
#############################################################
print("Find the probability: P(up, up, unchanged, down, unchanged, down, up|lamda")
probb =Probtill1+ Probtill2 + Probtill3
print(probb)
##########################################3
print("Fnd the optimal state sequence of the model which generates the observation sequence: (up, up, unchanged, down, unchanged, down, up)")
if (LargeProb1 > LargeProb2) and (LargeProb1 > LargeProb3):
	SEQ= Memory1
	large=LargeProb1
elif (LargeProb2 > LargeProb1) and (LargeProb2 > LargeProb3):
	SEQ=Memory2
	large=LargeProb2
else:
	SEQ=Memory3
	large=LargeProb3
##################################################################
print(SEQ)
print("Largest sequence probability: {}".format(large))
