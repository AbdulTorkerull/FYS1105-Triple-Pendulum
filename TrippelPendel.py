import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

dt = 0.01
k = 500

#Initial verdiene
g = 9.81
m1, m2, m3 = 1, 1, 1
l1, l2, l3 = 1, 1, 1
p1, p2, p3 = 0.2, 0.2, 0.8

p1_list = np.zeros(k); p1_list[0] = p1
p2_list = np.zeros(k); p2_list[0] = p2
p3_list = np.zeros(k); p3_list[0] = p3

pd1, pd2, pd3 = 0, 0, 0

pd1_list = np.zeros(k); pd1_list[0] = pd1
pd2_list = np.zeros(k); pd2_list[0] = pd2
pd3_list = np.zeros(k); pd3_list[0] = pd3

pdd1, pdd2, pdd3 = 0, 0, 0
xaxis = np.zeros(k)

#Analytisk løsning

def p(t, p1, p2, p3, g, l):
    c1 = np.array([0.332817, 0.287015, 0.18166])
    c2 = np.array([0.333192, 0.07838, -0.255345])
    c3 = np.array([0.333991, -0.365395, 0.0846849])
    h1 = np.array([1, 1.29211, 1.63122])
    h2 = np.array([1, 0.35826, -2.39812])
    h3 = np.array([1, -1.64497, 0.766897])
    p_list = np.array([p1, p2, p3])
    e1 = np.dot(p_list,c1)*h1*np.cos(np.sqrt(0.41577)*(g/l)*t)
    e2 = np.dot(p_list,c2)*h2*np.cos(np.sqrt(2.29428)*(g/l)*t)
    e3 = np.dot(p_list,c3)*h3*np.cos(np.sqrt(6.28994)*(g/l)*t)
    return e1 + e2 + e3

e1_list = np.zeros(k)
e2_list = np.zeros(k)
e3_list = np.zeros(k)
x_list = np.linspace(0,k*dt,k)
for i in range(k):
    e1_list[i] = p(i*dt, p1, p2, p3, g, l1)[0]
    e2_list[i] = p(i*dt, p1, p2, p3, g, l1)[1]
    e3_list[i] = p(i*dt, p1, p2, p3, g, l1)[2]

#Numerisk løsning
for i in range(k - 1):

    M = np.array([[(m1 + m2 + m3)*l1**2, (m2 + m3)*l1*l2*np.cos(p2 - p1), m3*l1*l3*np.cos(p3 - p1)]
        ,[(m2 + m3)*l1*l2*np.cos(p2 - p1), (m2 + m3)*l2**2, m3*l2*l3*np.cos(p3 - p2)]
        ,[m3*l1*l2*np.cos(p3 - p1), m3*l2*l3*np.cos(p3 - p2), m3*l3**2]])

    R = np.array([-(m1 + m2 + m3)*g*l1*np.sin(p1) + (m2 + m3)*l1*l2*np.sin(p2 - p1)*pd2**2 + m3*l1*l3*np.sin(p3 - p1)*pd3**2
        ,-(m2 + m3)*g*l2*np.sin(p2) - (m2 + m3)*l1*l2*np.sin(p2 - p1)*pd1**2 + m3*l2*l3*np.sin(p3 - p2)*pd3**2
        ,-m3*g*l3*np.sin(p3) - m3*l1*l3*np.sin(p3 - p1)*pd1**2 - m3*l2*l3*np.sin(p3 - p2)*pd2**2])
    
    #Finner den inverse til matrisen M
    pdd = np.linalg.solve(M,R)
    pdd1, pdd2, pdd3 = pdd[0], pdd[1], pdd[2]
    pd1_list[i + 1], pd2_list[i + 1], pd3_list[i + 1] = pd1_list[i] + pdd1*dt, pd2_list[i] + pdd2*dt, pd3_list[i] + pdd3*dt
    p1_list[i + 1], p2_list[i + 1], p3_list[i + 1] = p1_list[i] + pd1_list[i + 1]*dt, p2_list[i] + pd2_list[i + 1]*dt, p3_list[i] + pd3_list[i + 1]*dt
    xaxis[i + 1] = i*dt + dt

    pd1, pd2, pd3 = pd1_list[i + 1], pd2_list[i + 1], pd3_list[i + 1]
    p1, p2, p3 = p1_list[i + 1], p2_list[i + 1], p3_list[i + 1]



plt.subplot(3,1,1)
plt.plot(x_list,p1_list,label='Pendel 1')
plt.plot(x_list,e1_list,label='Pendel 1')
plt.legend()
plt.subplot(3,1,2)
plt.plot(xaxis,p2_list,label='Pendel 2')
plt.plot(xaxis,e2_list,label='Pendel 2')
plt.legend()
plt.subplot(3,1,3)
plt.plot(xaxis,p3_list,label='Pendel 3')
plt.plot(xaxis,e3_list,label='Pendel 3')
plt.legend()
plt.show()



l = (l1 + l2 + l3)*1.5
for i in range(k):
    plt.cla()
    plt.axis([-l,l,-l,l])
    plt.plot([0,0],[0,0],'ko')
    plt.plot(np.sin(e1_list[i])*l1,-np.cos(e1_list[i])*l1, 'bo')
    plt.plot(np.sin(e2_list[i])*(l1 + l2),-np.cos(e2_list[i])*(l1 + l2), 'ro')
    plt.plot(np.sin(e3_list[i])*(l1 + l2 + l3),-np.cos(e3_list[i])*(l1 + l2 + l3), 'go')

    plt.plot([0,np.sin(e1_list[i])*l1],[0,-np.cos(e1_list[i])*l1],linestyle='-')
    plt.plot([np.sin(e1_list[i])*l1,np.sin(e2_list[i])*(l1 + l2)],[-np.cos(e1_list[i])*l1,-np.cos(e2_list[i])*(l1 + l2)],linestyle='-')
    plt.plot([np.sin(e2_list[i])*(l1 + l2),np.sin(e3_list[i])*(l1 + l2 + l3)],[-np.cos(e2_list[i])*(l1 + l2),-np.cos(e3_list[i])*(l1 + l2 + l3)],linestyle='-')
    
    plt.pause(0.001)





