import numpy as np
############################################################
train1=[['A','B','B','C','A','B','C','A','A','B','C'],['A','B','C','A','B','C'],['A','B','C','A','A','B','C'],['B','B','A','B','C','A','B'],['B','C','A','A','B','C','C','A','B'],['C','A','C','C','A','B','C','A'],['C','A','B','C','A','B','C','A'],['C','A','B','C','A'],['C','A','B','C','A']]
train2=[['B','B','B','C','C','B','C'],['C','C','B','A','B','B'],['A','A','C','C','B','B','B'],['B','B','A','B','B','A','C'],['C','C','A','A','B','B','A','B'],['B','B','B','C','C','B','A','A'],['A','B','B','B','B','A','B','A'],['C','C','C','C','C'],['B','B','A','A','A']]
###########################################################
test1=['A','B','C','A','B','C','C','A','B']
test2=['A','A','B','A','B','C','C','C','C','B','B','B']
###################################################



class BW:
	def __init__(self,eventprob, maxiter):
		self.maxiter = maxiter
		###########################################################
		#initial setting
		#transition
		self.TransPTo1From=dict()
		self.TransPTo2From=dict()
		self.TransPTo3From=dict()
		##############################
		self.TransPTo1From['1'] =0.34
		self.TransPTo1From['2'] =0.33
		self.TransPTo1From['3'] =0.33

		self.TransPTo2From['1'] =0.33
		self.TransPTo2From['2'] =0.34
		self.TransPTo2From['3'] =0.33

		self.TransPTo3From['1'] =0.33
		self.TransPTo3From['2'] =0.33
		self.TransPTo3From['3'] =0.34

		#A B C
		self.EventP1 =dict()
		self.EventP2 =dict()
		self.EventP3 =dict()

		self.EventP1['A'] = eventprob[0][0]#0.34
		self.EventP1['B'] = eventprob[0][1]#0.33
		self.EventP1['C'] = eventprob[0][2]#0.33

		self.EventP2['A'] = eventprob[1][0]#0.33
		self.EventP2['B'] = eventprob[1][1]#0.34
		self.EventP2['C'] = eventprob[1][2]#0.33

		self.EventP3['A'] = eventprob[2][0]#0.33
		self.EventP3['B'] = eventprob[2][1]#0.33
		self.EventP3['C'] = eventprob[2][2]#0.34

		self.pi1=1/3
		self.pi2=1/3
		self.pi3=1/3
		#########################################################
		############################################################
	
	def train(self,seqs):
		L=len(seqs)
		
		for i in range(self.maxiter):

			########################################
			totalpi1=0
			totalpi2=0
			totalpi3=0
			#
			totala11num=0
			totala12num=0
			totala13num=0

			totala21num=0
			totala22num=0
			totala23num=0

			totala31num=0
			totala32num=0
			totala33num=0

			totala1de=0
			totala2de=0
			totala3de=0
			#
			totalep11num=0
			totalep12num=0
			totalep13num=0

			totalep21num=0
			totalep22num=0
			totalep23num=0

			totalep31num=0
			totalep32num=0
			totalep33num=0

			totalep1de=0
			totalep2de=0
			totalep3de=0


			############################################
			for nob in range(L):
				seq = seqs[nob]
				T = len(seq)
				N=3
				#####################################################################################
				#forward
				alpha = np.zeros((N, T))
				ob =seq[0]
				alpha[0][0] = self.pi1 * self.EventP1[ob]
				alpha[1][0] = self.pi2 * self.EventP2[ob]
				alpha[2][0] = self.pi3 * self.EventP3[ob]
				for t in range(1, T):
					ob = seq[t]
					##############3
					alpha[0][t] = (alpha[0][(t-1)] *self.TransPTo1From['1'] + alpha[1][(t-1)] *self.TransPTo1From['2'] +  alpha[2][(t-1)] *self.TransPTo1From['3']) * self.EventP1[ob]
					alpha[1][t] = (alpha[0][(t-1)] *self.TransPTo2From['1'] + alpha[1][(t-1)] *self.TransPTo2From['2'] +  alpha[2][(t-1)] *self.TransPTo2From['3']) * self.EventP2[ob]
					alpha[2][t] = (alpha[0][(t-1)] *self.TransPTo3From['1'] + alpha[1][(t-1)] *self.TransPTo3From['2'] +  alpha[2][(t-1)] *self.TransPTo3From['3']) * self.EventP3[ob]
				#####################################################################################
				#backward
				beta = np.zeros((N, T))
				beta[0][(T-1)] =1
				beta[1][(T-1)] =1
				beta[2][(T-1)] =1

				for t in range(T-2, -1, -1):
					ob = seq[(t+1)]
					beta[0][t] = (beta[0][(t+1)] *self.TransPTo1From['1'] *self.EventP1[ob]  + beta[1][(t+1)]*self.TransPTo2From['1']*self.EventP2[ob]   + beta[2][(t+1)]*self.TransPTo3From['1']* self.EventP3[ob])
					beta[1][t] = (beta[0][(t+1)] *self.TransPTo1From['2'] *self.EventP1[ob]  + beta[1][(t+1)]*self.TransPTo2From['2']*self.EventP2[ob]   + beta[2][(t+1)]*self.TransPTo3From['2']* self.EventP3[ob])
					beta[2][t] = (beta[0][(t+1)] *self.TransPTo1From['3'] *self.EventP1[ob]  + beta[1][(t+1)]*self.TransPTo2From['3']*self.EventP2[ob]   + beta[2][(t+1)]*self.TransPTo3From['3']* self.EventP3[ob])
				###################################################################################
				#Et(i,j)
				Etij = np.zeros((T-1, N, N))

				for t in range((T-1)):
					###################################
					temp=np.zeros((N,N))
					denom=0
					ob=seq[(t+1)]
					##################################################
					for m in range(N):
						#rr=0
						for n in range(N):
							##############################################3
							if m ==0:
								if n==0:
									amn = self.TransPTo1From['1']
									bn=self.EventP1[ob]
								elif n==1:
									amn = self.TransPTo2From['1']
									bn=self.EventP2[ob]
								else:
									amn = self.TransPTo3From['1']
									bn=self.EventP3[ob]
							elif m==1:
								if n==0:
									amn = self.TransPTo1From['2']
									bn=self.EventP1[ob]
								elif n==1:
									amn = self.TransPTo2From['2']
									bn=self.EventP2[ob]
								else:
									amn = self.TransPTo3From['2']
									bn=self.EventP3[ob]
							else:
								if n==0:
									amn = self.TransPTo1From['3']
									bn=self.EventP1[ob]
								elif n==1:
									amn = self.TransPTo2From['3']
									bn=self.EventP2[ob]
								else:
									amn = self.TransPTo3From['3']
									bn=self.EventP3[ob]
							##########################################################

							see=alpha[m][t]*amn*bn*beta[n][(t+1)]
							denom+=see
							temp[m][n]=see
					############################################################
					Etij[t] = temp/denom
					############################################################
				################################################################################
				
				#rti
				#######################################################
				#rti=np.zeros((N,T-1))
				#for t in range((T-1)):
				#	temp2=Etij[t]
					###############################
				#	for m in range(N):
						######################
				#		rr=0
				#		for n in range(N):
				#			rr+=temp2[m][n]
						#######################
				#		rti[m][t] =rr
				###############################3########################
				rti=np.zeros((N,T))

				for t in range(T):
					rde=alpha[0][t]*beta[0][t] + alpha[1][t]* beta[1][t] + alpha[2][t]*beta[2][t]

					rti[0][t] = (alpha[0][t]*beta[0][t])/rde 
					rti[1][t] = (alpha[1][t]*beta[1][t])/rde
					rti[2][t] = (alpha[2][t]*beta[2][t])/rde 
					
				########################################################
				#new aij
				a11num=0
				a12num=0
				a13num=0

				a21num=0
				a22num=0
				a23num=0

				a31num=0
				a32num=0
				a33num=0

				a1de=0
				a2de=0
				a3de=0
				for t in range((T-1)):
					a1de+=rti[0][t]
					a2de+=rti[1][t]
					a3de+=rti[2][t]

					temp3=Etij[t]
					a11num += temp3[0][0]
					a12num += temp3[0][1]
					a13num += temp3[0][2]

					a21num += temp3[1][0]
					a22num += temp3[1][1]
					a23num += temp3[1][2]

					a31num += temp3[2][0]
					a32num += temp3[2][1]
					a33num += temp3[2][2]
				###########################################3
				totala11num+=a11num
				totala12num+=a12num
				totala13num+=a13num

				totala21num+=a21num
				totala22num+=a22num
				totala23num+=a23num

				totala31num+=a31num
				totala32num+=a32num
				totala33num+=a33num

				totala1de+=a1de
				totala2de+=a2de
				totala3de+=a3de

				#############################################################3
				#new pi
				totalpi1+=rti[0][0]
				totalpi2+=rti[1][0]
				totalpi3+=rti[2][0]
				######################################################

				#bjvk
				ep11num=0#self.EventP1['A'] = 0.34
				ep12num=0#self.EventP1['B'] = 0.33
				ep13num=0#self.EventP1['C'] = 0.33

				ep21num=0#self.EventP2['A'] = 0.33
				ep22num=0#self.EventP2['B'] = 0.34
				ep23num=0#self.EventP2['C'] = 0.33

				ep31num=0#self.EventP3['A'] = 0.33
				ep32num=0#self.EventP3['B'] = 0.33
				ep33num=0#self.EventP3['C'] = 0.34

				ep1de=0
				ep2de=0
				ep3de=0

				for t in range(T):
					ob=seq[t]

					ep1de +=rti[0][t]
					ep2de +=rti[1][t]
					ep3de +=rti[2][t]
					#############################
					if ob == 'A':
						ep11num+=rti[0][t]
						ep21num+=rti[1][t]
						ep31num+=rti[2][t]
					elif ob == 'B':
						ep12num+=rti[0][t]
						ep22num+=rti[1][t]
						ep32num+=rti[2][t]
					else:
						ep13num+=rti[0][t]
						ep23num+=rti[1][t]
						ep33num+=rti[2][t]
				################################


				totalep11num+=ep11num
				totalep12num+=ep12num
				totalep13num+=ep13num

				totalep21num+=ep21num
				totalep22num+=ep22num
				totalep23num+=ep23num

				totalep31num+=ep31num
				totalep32num+=ep32num
				totalep33num+=ep33num

				totalep1de+=ep1de
				totalep2de+=ep2de
				totalep3de+=ep3de
				
				##############################################################
			######################################################################
			
			#######################################################################3
			#new pi
			self.pi1=(totalpi1)/L
			self.pi2=(totalpi2)/L
			self.pi3=(totalpi3)/L

			#########################################################3
			#aij
			self.TransPTo1From['1'] =totala11num/totala1de
			self.TransPTo1From['2'] =totala21num/totala2de
			self.TransPTo1From['3'] =totala31num/totala3de

			self.TransPTo2From['1'] =totala12num/totala1de
			self.TransPTo2From['2'] =totala22num/totala2de
			self.TransPTo2From['3'] =totala32num/totala3de

			self.TransPTo3From['1'] =totala13num/totala1de
			self.TransPTo3From['2'] =totala23num/totala2de
			self.TransPTo3From['3'] =totala33num/totala3de
			############################################################
			#bjvk
			self.EventP1['A'] = totalep11num/totalep1de
			self.EventP1['B'] = totalep12num/totalep1de
			self.EventP1['C'] = totalep13num/totalep1de

			self.EventP2['A'] = totalep21num/totalep2de
			self.EventP2['B'] = totalep22num/totalep2de
			self.EventP2['C'] = totalep23num/totalep2de

			self.EventP3['A'] = totalep31num/totalep3de
			self.EventP3['B'] = totalep32num/totalep3de
			self.EventP3['C'] = totalep33num/totalep3de

			#print('a1j')
			#print(self.TransPTo1From['1']+self.TransPTo2From['1']+self.TransPTo3From['1'])
			#print('a2j')
			#print(self.TransPTo1From['2']+self.TransPTo2From['2']+self.TransPTo3From['2'])
			#print('a3j')
			#print(self.TransPTo1From['3']+self.TransPTo2From['3']+self.TransPTo3From['3'])
			
			#print('pi')
			#print(self.pi1+self.pi2+self.pi3)
			#print('p1')
			#print(self.EventP1['A']+self.EventP1['B']+self.EventP1['C'])
			#print('p2')
			#print(self.EventP2['A']+self.EventP2['B']+self.EventP2['C'])
			#print('p3')
			#print(self.EventP3['A']+self.EventP3['B']+self.EventP3['C'])
		##################################################################################

	def getA(self):
		A =np.zeros((3,3))
		A[0][0]=self.TransPTo1From['1']
		A[0][1]=self.TransPTo2From['1']
		A[0][2]=self.TransPTo3From['1']

		A[1][0]=self.TransPTo1From['2']
		A[1][1]=self.TransPTo2From['2']
		A[1][2]=self.TransPTo3From['2']

		A[2][0]=self.TransPTo1From['3']
		A[2][1]=self.TransPTo2From['3']
		A[2][2]=self.TransPTo3From['3']

		return A

	def getB(self):
		B = np.zeros((3,3))

		B[0][0]=self.EventP1['A']
		B[0][1]=self.EventP1['B']
		B[0][2]=self.EventP1['C']

		B[1][0]=self.EventP2['A']
		B[1][1]=self.EventP2['B']
		B[1][2]=self.EventP2['C']

		B[2][0]=self.EventP3['A']
		B[2][1]=self.EventP3['B']
		B[2][2]=self.EventP3['C']

		return B

	def getPi(self):

		return np.array([self.pi1,self.pi2,self.pi3])


	def test(self, seqs2):
		T2=len(seqs2)
		########################################################
		#initialization
		#########################################################
		ob=seqs2[0]
		Probtill1=self.pi1 *self.EventP1[ob]
		Probtill2=self.pi2 *self.EventP2[ob]
		Probtill3=self.pi3 *self.EventP3[ob]

		#to record the largest path prob
		LargeProb1=self.pi1 *self.EventP1[ob]
		LargeProb2=self.pi2 *self.EventP2[ob]
		LargeProb3=self.pi3 *self.EventP3[ob]

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
		for i in range(1, T2):
			#print("round: {}".format(i))
			newob =seqs2[i]
			##############################################################################################################
			#Discuss STATE1
			#PART1: for total prob 
			P11=Probtill1 * self.TransPTo1From['1']
			P12=Probtill2 * self.TransPTo1From['2']
			P13=Probtill3 * self.TransPTo1From['3']

			newProbtill1 = (P11+P12+P13) * self.EventP1[newob]

			#PART2: for largest prob
			p11=LargeProb1 * self.TransPTo1From['1']
			p12=LargeProb2 * self.TransPTo1From['2']
			p13=LargeProb3 * self.TransPTo1From['3']

			if (p11 > p12) and (p11> p13):
				newMemory1 =Memory1.copy()
				newMemory1.append('1')

				newLargeProb1 = p11* self.EventP1[newob]
			elif (p12 > p11) and (p12>p13):
				newMemory1 =Memory2.copy()
				newMemory1.append('1')

				newLargeProb1 = p12* self.EventP1[newob]
			else:
				newMemory1 =Memory3.copy()
				newMemory1.append('1')

				newLargeProb1 = p13* self.EventP1[newob]

	
			#####################################################
			#Discuss STATE2
			#PART1: for total prob 

			P21=Probtill1 * self.TransPTo2From['1']
			P22=Probtill2 * self.TransPTo2From['2']
			P23=Probtill3 * self.TransPTo2From['3']

			newProbtill2 = (P21+P22+P23) * self.EventP2[newob]

			#PART2: for largest prob
			p21=LargeProb1 * self.TransPTo2From['1']
			p22=LargeProb2 * self.TransPTo2From['2']
			p23=LargeProb3 * self.TransPTo2From['3']

			if (p21 > p22) and (p21> p23):
				newMemory2 =Memory1.copy()
				newMemory2.append('2')

				newLargeProb2 = p21* self.EventP2[newob]
			elif (p22 > p21) and (p22>p23):
				newMemory2 =Memory2.copy()
				newMemory2.append('2')

				newLargeProb2 = p22* self.EventP2[newob]
			else:
				newMemory2 =Memory3.copy()
				newMemory2.append('2')

				newLargeProb2 = p23* self.EventP2[newob]

			#########################################################
			#Discuss STATE3
			#PART1: for total prob 

			P31=Probtill1 * self.TransPTo3From['1']
			P32=Probtill2 * self.TransPTo3From['2']
			P33=Probtill3 * self.TransPTo3From['3']

			newProbtill3 = (P31+P32+P33) * self.EventP3[newob]

			#PART2: for largest prob
			p31=LargeProb1 * self.TransPTo3From['1']
			p32=LargeProb2 * self.TransPTo3From['2']
			p33=LargeProb3 * self.TransPTo3From['3']

			if (p31 > p32) and (p31> p33):
				newMemory3 =Memory1.copy()
				newMemory3.append('3')

				newLargeProb3 = p31* self.EventP3[newob]
			elif (p32 > p31) and (p32>p33):
				newMemory3 =Memory2.copy()
				newMemory3.append('3')

				newLargeProb3 = p32* self.EventP3[newob]
			else:
				newMemory3 =Memory3.copy()
				newMemory3.append('3')

				newLargeProb3 = p33* self.EventP3[newob]
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
		##################################################################
		##################################################################
		#print("Find the probability: P(up, up, unchanged, down, unchanged, down, up|lamda")
		probb =Probtill1+ Probtill2 + Probtill3
		#print(probb)
		##########################################3
		#print("Fnd the optimal state sequence of the model which generates the observation sequence: (up, up, unchanged, down, unchanged, down, up)")
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
		#print(SEQ)
		#print("Largest sequence probability: {}".format(large))
		return probb







eventprob1= np.array([[0.34,0.33,0.33],[0.33,0.34,0.33],[0.33,0.33,0.34]])
eventprob2= np.array([[1,0,0],[0,1,0],[0,0,1]])
###########################################################
#model1=BW(eventprob=eventprob1 , maxiter=50)
#model1.train(train1)
#prob11=model1.test(test1)
#prob12=model1.test(test2)
###############################################################
#model2=BW(eventprob=eventprob1 , maxiter=50)
#model2.train(train2)
#prob21=model2.test(test1)
#prob22=model2.test(test2)
##########################################################33
print("P1-1: Please specify the model parameters after the first iterations of Baum-Welch training:")
print("Trainset1:")
model1=BW(eventprob=eventprob1 , maxiter=1)
model1.train(train1)
print("Transition probability:")
print(model1.getA())
print("Observation probability:")
print(model1.getB())
print("Initial state probability:")
print(model1.getPi())
print('------------------------------------------')
print("Trainset2:")
model2=BW(eventprob=eventprob1 , maxiter=1)
model2.train(train2)
print("Transition probability:")
print(model2.getA())
print("Observation probability:")
print(model2.getB())
print("Initial state probability:")
print(model2.getPi())
print('################################################################################')
print("P1-2: Please specify the model parameters after 50th iterations of Baum-Welch training")
print("Trainset1:")
model3=BW(eventprob=eventprob1 , maxiter=50)
model3.train(train1)
print("Transition probability:")
print(model3.getA())
print("Observation probability:")
print(model3.getB())
print("Initial state probability:")
print(model3.getPi())
print('------------------------------------------')
print("Trainset2:")
model4=BW(eventprob=eventprob1 , maxiter=50)
model4.train(train2)
print("Transition probability:")
print(model4.getA())
print("Observation probability:")
print(model4.getB())
print("Initial state probability:")
print(model4.getPi())
print("################################################################################################################")


correct1=0
wrong1=0

correct50=0
wrong50=0

for tes in train1:
	prob1=model1.test(tes)
	prob2=model2.test(tes)
	
	if prob1 > prob2:
		correct1 +=1
	else:
		wrong1 +=1

for tes2 in train2:
	prob1=model1.test(tes2)
	prob2=model2.test(tes2)
	
	if prob1 < prob2:
		correct1 +=1
	else:
		wrong1 +=1

acc1=correct1/(correct1 + wrong1)
#####################################
for tes3 in train1:
	prob1=model3.test(tes3)
	prob2=model4.test(tes3)
	
	if prob1 > prob2:
		correct50 +=1
	else:
		wrong50 +=1

for tes4 in train2:
	prob1=model3.test(tes4)
	prob2=model4.test(tes4)
	
	if prob1 < prob2:
		correct50 +=1
	else:
		wrong50 +=1

acc50=correct50/(correct50 + wrong50)
#################################################
print("P2: Please show the recognition results by using the above training sequences as the testing data (The so-called inside testing).")
print("First iteration:")
print("Accuracy: {} %".format(acc1*100))
print('-----------------------------')
print("50th iteration:")
print("Accuracy: {} %".format(acc50*100))
###################################################

print("##################################################################################")

print("P3. Which class do the following testing sequences belong to?")
print("Using 50th iteration models.")
print("test data 1: ABCABCCAB")
prob1=model3.test(test1)
prob2=model4.test(test1)

if prob1 > prob2:
	print("test data 1 is model 1")
else:
	print("test data 1 is model 2")
#print(prob11)
#print(prob21)
print("--------------------------------------------")
############################################################

print("test data 2:  AABABCCCCBBB")
prob1=model3.test(test2)
prob2=model4.test(test2)
if prob1 > prob2:
	print("test data 2 is model 1")
else:
	print("test data 2 is model 2")

#print(prob12)
#print(prob22)
#############################################################





