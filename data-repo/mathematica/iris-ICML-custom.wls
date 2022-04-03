#!/usr/bin/env wolframscript

Get["./CRNSimulator.m"];

IRIS[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x0Sneg, x0SnegInitial],
conc[x1Spos, x1SposInitial],
conc[x1Sneg, x1SnegInitial],
conc[x2Spos, x2SposInitial],
conc[x2Sneg, x2SnegInitial],
conc[x3Spos, x3SposInitial],
conc[x3Sneg, x3SnegInitial],
rxn[x0Spos,iL0N0Sneg+iL0N1Sneg+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N5Spos+iL0N6Sneg+iL0N7Sneg,1],
rxn[x0Sneg,iL0N0Spos+iL0N1Spos+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Sneg+iL0N6Spos+iL0N7Spos,1],
rxn[x1Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Spos+iL0N4Sneg+iL0N5Sneg+iL0N6Spos+iL0N7Spos,1],
rxn[x1Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Sneg+iL0N7Sneg,1],
rxn[x2Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N5Spos+iL0N6Sneg+iL0N7Sneg,1],
rxn[x2Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Sneg+iL0N6Spos+iL0N7Spos,1],
rxn[x3Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos+iL0N4Spos+iL0N5Spos+iL0N6Sneg+iL0N7Sneg,1],
rxn[x3Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Sneg+iL0N5Sneg+iL0N6Spos+iL0N7Spos,1],
conc[iL0N0Sneg,1.7928376126],
conc[iL0N1Sneg,2.2293605977],
conc[iL0N2Sneg,3.2474348903],
conc[iL0N3Sneg,2.8792841132],
conc[iL0N4Sneg,2.9100382550],
conc[iL0N5Sneg,1.7720156302],
conc[iL0N6Sneg,0.6535084321],
conc[iL0N7Sneg,1.4854952560],
rxn[iL0N0Spos,mL0N0+hL0N0Spos,1],
rxn[mL0N0+iL0N0Sneg,hL0N0Sneg,1],
rxn[iL0N1Spos,mL0N1+hL0N1Spos,1],
rxn[mL0N1+iL0N1Sneg,hL0N1Sneg,1],
rxn[iL0N2Spos,mL0N2+hL0N2Spos,1],
rxn[mL0N2+iL0N2Sneg,hL0N2Sneg,1],
rxn[iL0N3Spos,mL0N3+hL0N3Spos,1],
rxn[mL0N3+iL0N3Sneg,hL0N3Sneg,1],
rxn[iL0N4Spos,mL0N4+hL0N4Spos,1],
rxn[mL0N4+iL0N4Sneg,hL0N4Sneg,1],
rxn[iL0N5Spos,mL0N5+hL0N5Spos,1],
rxn[mL0N5+iL0N5Sneg,hL0N5Sneg,1],
rxn[iL0N6Spos,mL0N6+hL0N6Spos,1],
rxn[mL0N6+iL0N6Sneg,hL0N6Sneg,1],
rxn[iL0N7Spos,mL0N7+hL0N7Spos,1],
rxn[mL0N7+iL0N7Sneg,hL0N7Sneg,1],
rxn[hL0N0Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N0Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N1Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N1Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N2Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N2Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N3Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N3Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N4Spos,hL1N0Sneg+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N4Sneg,hL1N0Spos+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N5Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N5Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N6Spos,hL1N0Spos+hL1N1Sneg+hL1N2Sneg,1],
rxn[hL0N6Sneg,hL1N0Sneg+hL1N1Spos+hL1N2Spos,1],
rxn[hL0N7Spos,hL1N0Spos+hL1N1Sneg+hL1N2Sneg,1],
rxn[hL0N7Sneg,hL1N0Sneg+hL1N1Spos+hL1N2Spos,1],
conc[hL1N0Sneg,0.9570537447],
conc[hL1N1Spos,0.5652184891],
conc[hL1N2Sneg,0.5435687859]
}

IRISReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_] =
  reduceFFNCCrn[IRIS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], True];

cnt = Count[IRIS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[IRISReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], rxn[___]]
Print["Number of reactions (reduced form): " <> ToString[cntReduced]];

tmax=20;
totalCount=0;
correctCount=0;
matchingCount=0;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=IRIS[0,0.900681,1.032057,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
For[i=0,i<=tmax,i++, Print["y0(" <> ToString[i] <> ")=" <> ToString[(hL1N0Spos[i] - hL1N0Sneg[i])/.sol]]];
For[i=0,i<=tmax,i++, Print["y1(" <> ToString[i] <> ")=" <> ToString[(hL1N1Spos[i] - hL1N1Sneg[i])/.sol]]];
For[i=0,i<=tmax,i++, Print["y2(" <> ToString[i] <> ")=" <> ToString[(hL1N2Spos[i] - hL1N2Sneg[i])/.sol]]];
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "*"];
Print["0: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0 (setosa)","y1 (versicolor)","y2 (virginica)"}, PlotRange->{{0,20},{-10,7}}, AxesLabel->{"time","value"}];
Export["../results/IRIS_p0-custom.pdf", p0];

Print["i: E C N"];
rsys=normalizeCRN[IRISReduced[0,0.900681,1.032057,0,0,1.341272,0,1.312977]];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
For[i=0,i<=tmax,i++, Print["y0(" <> ToString[i] <> ")=" <> ToString[(hL1N0Spos[i] - hL1N0Sneg[i])/.sol]]];
For[i=0,i<=tmax,i++, Print["y1(" <> ToString[i] <> ")=" <> ToString[(hL1N1Spos[i] - hL1N1Sneg[i])/.sol]]];
For[i=0,i<=tmax,i++, Print["y2(" <> ToString[i] <> ")=" <> ToString[(hL1N2Spos[i] - hL1N2Sneg[i])/.sol]]];
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "*"];
Print["0: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0 (setosa)","y1 (versicolor)","y2 (virginica)"}, PlotRange->{{0,20},{-10,7}}, AxesLabel->{"time","value"}];
Export["../results/IRISReduced_p0-custom.pdf", p0];