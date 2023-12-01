import ROOT
import numpy as np
import math as m
import matplotlib.pyplot as plt
import os
import tqdm

ROOT.gROOT.SetBatch(True)

def nparr(list):
    return np.array(list, dtype="d")
def fill_h(histo_name, array):
    for x in range (len(array)):
        histo_name.Fill((np.array(array[x] ,dtype="d")))
def hist(list, x_name, channels=1000, linecolor=4, linewidth=4,write=True):
    array=np.array(list ,dtype="d")
    hist=ROOT.TH1D(x_name,x_name,channels,0.99*np.min(array),1.01*np.max(array))
    fill_h(hist,array)
    hist.SetLineColor(linecolor)
    hist.SetLineWidth(linewidth)
    hist.GetXaxis().SetTitle(x_name)
    hist.GetYaxis().SetTitle("Entries")
    if write==True: hist.Write()
    #hist.SetStats(False)
    hist.GetYaxis().SetMaxDigits(3);
    hist.GetXaxis().SetMaxDigits(3);
    return hist
def grapherr(x,y,ex,ey,x_string, y_string,name=None, color=4, markerstyle=22, markersize=2,write=True):
        plot = ROOT.TGraphErrors(len(x),  np.array(x  ,dtype="d")  ,   np.array(y  ,dtype="d") , np.array(   ex   ,dtype="d"),np.array( ey   ,dtype="d"))
        if name is None: plot.SetNameTitle(y_string+" vs "+x_string,y_string+" vs "+x_string)
        else: plot.SetNameTitle(name, name)
        plot.GetXaxis().SetTitle(x_string)
        plot.GetYaxis().SetTitle(y_string)
        plot.SetMarkerColor(color)#blue
        plot.SetMarkerStyle(markerstyle)
        plot.SetMarkerSize(markersize)
        if write==True: plot.Write()
        return plot

#root file creation
main=ROOT.TFile("3DanalUnPol.root","RECREATE")
main.mkdir("3D_UnPol")
def get_gamma(energy):
    """in keV"""
    return (energy+511)/511
def get_beta(energy):
    """in keV"""
    gamma=(energy+511)/511
    return(np.sqrt(1-(float(1.)/gamma)**2))

#plot beta and gamma as a function of the energy
tries=1000000
betas,gammas,energies=np.empty(tries),np.empty(tries),np.empty(tries)
for i in range(1,tries):
    energies[i]=i
    betas[i]=get_beta(i)
    gammas[i]=get_gamma(i)
grapherr(energies,betas,np.zeros(tries),np.zeros(tries),"Energy (keV)","Beta")
grapherr(energies,gammas,np.zeros(tries),np.zeros(tries),"Energy (keV)","gamma")

#plot prob sitribution as a funciton of theta Unpolarized fixed beta
def UnPolProbEmission(beta,theta):
    if theta==0:
        return 0
    if beta==1:
        raise ValueError("Input beta should be different than 1")
    sin_squared = m.sin(theta) ** 2
    denominator = (1 - beta*m.cos(theta)) ** 4

    result = 0.5*(sin_squared / denominator)
    return result

main.cd("3D_UnPol")

#point in the spherical plot
nPoints = 10000000
#generate angles randomly gen on a sphere
theta = 2 * m.pi * np.random.rand(nPoints)  # Azimuthal angle (longitude)
phi =  m.pi * np.random.rand(nPoints)  # Azimuthal angle (longitude)
#phi = np.arccos(2 * np.random.rand(nPoints) - 1)  # Polar angle (latitude)

if not os.path.exists("3D_UnPol"):
    os.makedirs("3D_UnPol")
else:
    os.system("rm 3D_UnPol/*")
#energy of the photon
energies=np.arange(0,75,1)
for n,en in tqdm.tqdm(enumerate(energies)):
    #compute the probabilities
    prob=np.empty(nPoints)
    for i,ang in enumerate(theta):
        prob[i]=UnPolProbEmission(get_beta(en),ang)
    # Create a 3D graph
    graph = ROOT.TGraph2D()
    graph.SetTitle(f"Angular distribution for {en}keV; X; Y; Z");
    if not os.path.exists("tempUNPOL"):
        os.makedirs("tempUNPOL")
    else:
        os.system("rm tempUNPOL/*")
    # Define the number of data points
    points=0
    """
    for i,th in enumerate(theta):
        for j,ph in enumerate(phi):
            x = prob[i]* np.sin(phi[j]) * np.cos(theta[i])
            y = prob[i] * np.sin(phi[j]) * np.sin(theta[i])
            z = prob[i]* np.cos(phi[j])
            # Add the point to the TGraph2
            graph.SetPoint(points, x, y, z)
            points=points+1
    """
    for i in range(nPoints):
        x = prob[i]* np.sin(phi[i]) * np.cos(theta[i])
        y = prob[i] * np.sin(phi[i]) * np.sin(theta[i])
        z = prob[i]* np.cos(phi[i])
        # Add the point to the TGraph2
        graph.SetPoint(points, x, y, z)
        points=points+1
    # Set the axis labels
    graph.GetXaxis().SetTitle("X")
    graph.GetYaxis().SetTitle("Y")
    graph.GetZaxis().SetTitle("Z")
    # save the spherical plot
    #graph.Write()
    #generate the view and the folder to store temporary pics
    th=[0,30,0,90]
    ph=[0,60,90,180]
    #cycle over the views
    for k in range(len(th)):
        canvas = ROOT.TCanvas("SphericalCanvas", "Spherical Plot", 800, 800)
        canvas.SetTheta(th[k])
        canvas.SetPhi(ph[k])
        # Show the canvas
        graph.Draw("pcol")  # pcol specifies a 3D scatter plot
        canvas.Update()
        canvas.SaveAs(f"tempUNPOL/test_{k:02}.png")
    os.system(f"./createIMG.sh tempUNPOL 3D_UnPol/energy_{n:02}keV.png")

os.system("./createGIF.sh 3D_UnPol energyScan3D_UnPOL.gif")