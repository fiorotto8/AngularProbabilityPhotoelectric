import ROOT
import numpy as np
import math as m
import matplotlib.pyplot as plt
import os

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
main=ROOT.TFile("2Danal.root","RECREATE")
main.mkdir("2D_UnPol")
def get_gamma(energy):
    """in keV"""
    return (energy+511)/511
def get_beta(energy):
    """in keV"""
    gamma=(energy+511)/511
    return(np.sqrt(1-(float(1.)/gamma)**2))

#plot beta and gamma as a function of the energy
tries=10000
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

#make 2D of Unpolorazied
energies=np.arange(0,100,1)
main.cd("2D_UnPol")
for j in range(len(energies)):
    angle=np.arange(0,2*m.pi,0.01)
    prob=np.empty(tries)
    for i,ang in enumerate(angle):
        prob[i]=UnPolProbEmission(get_beta(energies[j]),ang)
    grapherr(angle,prob,np.zeros(tries),np.zeros(tries),"Angle (rad)","Prob")

    # Create a TGraphPolar object
    graph_polar = ROOT.TGraphPolar()
    graph_polar.SetTitle("TGraphPolar Example")
    # Fill the TGraphPolar with data
    for i in range(len(angle)):
        graph_polar.SetPoint(i, angle[i], prob[i])
    graph_polar.Write()
    # Create a canvas
    canvas = ROOT.TCanvas("polar_canvas", "Polar Plot", 800, 800)
    # Draw the polar plot
    graph_polar.Draw("APL")  # A: Axis, P: Polar, L: Line
    graph_polar.SetTitle(f"Polar Plot @ {energies[j]:02} keV")
    graph_polar.SetLineWidth(4)
    graph_polar.SetLineColor(9)
    # Show the canvas
    canvas.Update()
    canvas.SaveAs(f"polplot/frame_{j:02}.png")
os.system("./createGIF.sh polplot energyScan2D.gif")