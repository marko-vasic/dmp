TODO: This is not actual file used in ICML. Use gitk to retrieve correct version.

#!/usr/bin/env wolframscript

Get["CRNSimulator.m"];

DISEASE[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x0Sneg, x0SnegInitial],
conc[x1Spos, x1SposInitial],
conc[x1Sneg, x1SnegInitial],
conc[x2Spos, x2SposInitial],
conc[x2Sneg, x2SnegInitial],
conc[x3Spos, x3SposInitial],
conc[x3Sneg, x3SnegInitial],
conc[x4Spos, x4SposInitial],
conc[x4Sneg, x4SnegInitial],
conc[x5Spos, x5SposInitial],
conc[x5Sneg, x5SnegInitial],
conc[x6Spos, x6SposInitial],
conc[x6Sneg, x6SnegInitial],
conc[x7Spos, x7SposInitial],
conc[x7Sneg, x7SnegInitial],
conc[x8Spos, x8SposInitial],
conc[x8Sneg, x8SnegInitial],
conc[x9Spos, x9SposInitial],
conc[x9Sneg, x9SnegInitial],
rxn[x0Spos,+iL0N1Spos+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Spos,1],
rxn[x0Sneg,+iL0N1Sneg+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N5Sneg+iL0N6Sneg,1],
rxn[x1Spos,iL0N0Sneg+iL0N1Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Sneg+iL0N6Spos,1],
rxn[x1Sneg,iL0N0Spos+iL0N1Spos+iL0N3Spos+iL0N4Sneg+iL0N5Spos+iL0N6Sneg,1],
rxn[x2Spos,iL0N0Sneg+iL0N1Spos+iL0N5Sneg+iL0N7Spos,1],
rxn[x2Sneg,iL0N0Spos+iL0N1Sneg+iL0N5Spos+iL0N7Sneg,1],
rxn[x3Spos,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N7Sneg,1],
rxn[x3Sneg,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N7Spos,1],
rxn[x4Spos,+iL0N1Spos+iL0N2Spos+iL0N4Sneg+iL0N5Spos,1],
rxn[x4Sneg,+iL0N1Sneg+iL0N2Sneg+iL0N4Spos+iL0N5Sneg,1],
rxn[x5Spos,iL0N0Sneg+iL0N2Spos+iL0N3Sneg+iL0N5Sneg+iL0N7Sneg,1],
rxn[x5Sneg,iL0N0Spos+iL0N2Sneg+iL0N3Spos+iL0N5Spos+iL0N7Spos,1],
rxn[x6Spos,+iL0N2Spos+iL0N4Spos+iL0N6Spos+iL0N7Sneg,1],
rxn[x6Sneg,+iL0N2Sneg+iL0N4Sneg+iL0N6Sneg+iL0N7Spos,1],
rxn[x7Spos,iL0N0Spos+iL0N2Spos+iL0N4Spos+iL0N5Spos+iL0N6Spos,1],
rxn[x7Sneg,iL0N0Sneg+iL0N2Sneg+iL0N4Sneg+iL0N5Sneg+iL0N6Sneg,1],
rxn[x8Spos,iL0N0Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Spos+iL0N7Sneg,1],
rxn[x8Sneg,iL0N0Spos+iL0N3Spos+iL0N4Sneg+iL0N5Sneg+iL0N6Sneg+iL0N7Spos,1],
rxn[x9Spos,+iL0N1Spos+iL0N2Spos+iL0N4Spos+iL0N6Spos,1],
rxn[x9Sneg,+iL0N1Sneg+iL0N2Sneg+iL0N4Sneg+iL0N6Sneg,1],
conc[iL0N0Sneg,1.2758526419],
conc[iL0N1Sneg,1.2969974197],
conc[iL0N2Sneg,0.3930018561],
conc[iL0N3Sneg,1.7513571313],
conc[iL0N4Sneg,2.5229493754],
conc[iL0N5Sneg,0.7467788462],
conc[iL0N6Sneg,2.2925508280],
conc[iL0N7Sneg,1.1753611723],
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
rxn[hL0N0Spos,hL1N0Spos+hL1N2Sneg+hL1N3Sneg,1],
rxn[hL0N0Sneg,hL1N0Sneg+hL1N2Spos+hL1N3Spos,1],
rxn[hL0N1Spos,hL1N0Sneg+hL1N2Spos+hL1N3Sneg,1],
rxn[hL0N1Sneg,hL1N0Spos+hL1N2Sneg+hL1N3Spos,1],
rxn[hL0N2Spos,hL1N0Sneg+hL1N2Sneg+hL1N3Spos,1],
rxn[hL0N2Sneg,hL1N0Spos+hL1N2Spos+hL1N3Sneg,1],
rxn[hL0N3Spos,hL1N0Spos+hL1N1Sneg+hL1N3Sneg,1],
rxn[hL0N3Sneg,hL1N0Sneg+hL1N1Spos+hL1N3Spos,1],
rxn[hL0N4Spos,+hL1N1Spos+hL1N2Sneg+hL1N3Sneg,1],
rxn[hL0N4Sneg,+hL1N1Sneg+hL1N2Spos+hL1N3Spos,1],
rxn[hL0N5Spos,hL1N0Sneg+hL1N2Spos+hL1N3Sneg,1],
rxn[hL0N5Sneg,hL1N0Spos+hL1N2Sneg+hL1N3Spos,1],
rxn[hL0N6Spos,hL1N0Sneg+hL1N1Spos+hL1N2Sneg+hL1N3Sneg,1],
rxn[hL0N6Sneg,hL1N0Spos+hL1N1Sneg+hL1N2Spos+hL1N3Spos,1],
rxn[hL0N7Spos,hL1N0Spos+hL1N1Sneg+hL1N2Sneg+hL1N3Sneg,1],
rxn[hL0N7Sneg,hL1N0Sneg+hL1N1Spos+hL1N2Spos+hL1N3Spos,1],
conc[hL1N0Sneg,0.8730862031],
conc[hL1N1Sneg,1.7619898791],
conc[hL1N2Spos,0.6523396022],
conc[hL1N3Sneg,0.5230450538]
}

DISEASEReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_] =
  reduceFFNCCrn[DISEASE[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial], True];

cnt = Count[DISEASE[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[DISEASEReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial], rxn[___]]
Print["Number of reactions (reduced form): " <> ToString[cntReduced]];

tmax=50;
totalCount=0;
correctCount=0;
matchingCount=0;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=DISEASE[0.652789,0,0,0.023067,0.461607,0,0,0.272034,0,0.438868,0,0.362966,0,0.623959,0,0.579211,0,0.149747,0,0.474706];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["0: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p0.pdf", p0];

rsys=DISEASE[0.880243,0,1.053501,0,1.529114,0,1.159462,0,1.181329,0,1.381449,0,1.932063,0,2.162209,0,0.964706,0,1.525284,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p1.pdf", p1];

rsys=DISEASE[1.061022,0,1.659602,0,0.588261,0,1.641145,0,1.047604,0,0,0.255142,0.301024,0,1.625134,0,0.265941,0,1.615191,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["2: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p2.pdf", p2];

rsys=DISEASE[0,1.197669,0,1.057072,0,0.964895,0,1.524228,0,1.636175,0,0.799803,0,0.445159,0,0.900400,0,0.944686,0,0.834337];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p3.pdf", p3];

rsys=DISEASE[0,0.762404,0,1.292778,0,1.832227,0,0.462179,0,0.325927,0.747530,0,0,1.192358,0,0.978716,1.050472,0,0,1.316903];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["4: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p4.pdf", p4];

rsys=DISEASE[0,0.867577,0,0.846525,0,1.041863,0,1.432318,0,1.212129,0,1.002422,0,0.411887,0,1.550701,0,1.033922,0,1.867359];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p5.pdf", p5];

rsys=DISEASE[0,1.597877,0,1.133118,0,0.937324,0,1.216430,0,0.506771,0,1.176376,0,1.434652,0,0.494291,0,0.298472,0,0.465531];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p6.pdf", p6];

rsys=DISEASE[0,0.289758,0,0.460996,0,0.933016,0,1.131520,0,0.364499,0,0.716085,0,0.562242,0,0.411636,0,1.713598,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p7.pdf", p7];

rsys=DISEASE[1.216459,0,0.146997,0,0,0.206410,0.251426,0,0,0.505615,0.178437,0,0.769104,0,0,0.623370,0,0.898334,0,0.228836];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["8: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p8.pdf", p8];

rsys=DISEASE[0,0.227034,0,0.704080,0,0.252074,0,0.796849,0,1.025974,0,1.745794,0,1.089244,0,0.411636,0,0.856443,0,0.762777];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["9: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p9.pdf", p9];

rsys=DISEASE[0.553952,0,0.300414,0,0.515887,0,0,0.064729,0.650349,0,0,0.049916,0.182164,0,0,0.073652,0.053265,0,0.323456,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["10: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p10.pdf", p10];

rsys=DISEASE[0.361556,0,0,0.255746,0.977986,0,0,0.109893,0,0.043231,0,0.115067,0,1.374966,0,1.078543,1.455253,0,0.139970,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["11: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p11.pdf", p11];

rsys=DISEASE[0,1.436315,0,0.786748,0,0.823595,0,0.806108,0,0.450186,0,0.908605,0,0.858379,0,1.164785,0,0.932044,0,1.968276];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p12.pdf", p12];

rsys=DISEASE[1.523531,0,1.979867,0,0.445238,0,0.516994,0,0,0.148087,0,0.319967,0.756151,0,1.136748,0,0.068385,0,1.433541,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["13: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p13.pdf", p13];

rsys=DISEASE[0.959861,0,0.703346,0,0.342421,0,0,0.087310,0.812022,0,0.766099,0,0.513858,0,0,0.358041,0.378727,0,0,0.228836];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["14: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p14.pdf", p14];

rsys=DISEASE[0.880243,0,0.695591,0,1.167248,0,2.156249,0,1.554564,0,0.973279,0,1.357822,0,2.162209,0,1.184325,0,2.061060,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p15.pdf", p15];

rsys=DISEASE[0.127345,0,0.370217,0,0.849323,0,0.468893,0,0.371810,0,1.103580,0,1.518587,0,0.405678,0,0,0.890649,0,0.476540];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["16: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p16.pdf", p16];

rsys=DISEASE[0.746136,0,0.127890,0,0,0.408596,0,0.606480,0.511540,0,0.276490,0,0.233213,0,0.608919,0,0.253547,0,0.875746,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["17: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p17.pdf", p17];

rsys=DISEASE[1.590691,0,3.880838,0,2.029985,0,2.975990,0,1.817862,0,1.821217,0,3.315473,0,0.260557,0,0,0.670039,0,0.082048];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["18: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p18.pdf", p18];

rsys=DISEASE[0,1.471585,0,0.372086,0,0.330479,0,2.997277,0,1.400824,0.079082,0,0,0.471826,0,2.486336,0,0.763736,0,2.124788];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p19.pdf", p19];

rsys=DISEASE[0,0.080889,0,0.666625,0,0.961449,0.051119,0,0,0.387362,0,1.327525,0,0.186103,0,0.411636,0,1.615688,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p20.pdf", p20];

rsys=DISEASE[1.712759,0,2.055157,0,1.045764,0,1.146591,0,0.403913,0,0.231210,0,0.225086,0,0.948791,0,0,0.894119,1.615191,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["21: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p21.pdf", p21];

rsys=DISEASE[0,0.613514,0,0.231533,0.247646,0,0.563061,0,1.046218,0,0.414609,0,0.656593,0,0,0.191409,0.560667,0,0.145475,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["22: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p22.pdf", p22];

rsys=DISEASE[0.331778,0,0.008335,0,0.390096,0,0,0.660000,0,0.227538,1.011392,0,0.447315,0,0,0.010811,0.112011,0,0,0.184799];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["23: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p23.pdf", p23];

rsys=DISEASE[0,0.787325,0,1.175492,0,0.341680,0.562159,0,0.357720,0,0.504844,0,0,0.328075,0,0.663378,0,0.147515,0,0.087551];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["24: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p24.pdf", p24];

rsys=DISEASE[0,1.197669,0,1.285400,0,0.992466,0,1.150264,0,0.318537,0,0.705660,0,0.525415,0,1.519376,0,0.401837,0,0.280212];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["25: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p25.pdf", p25];

rsys=DISEASE[0,0.289758,0,0.685352,0,0.677699,0,0.908407,0,3.192858,0,1.359450,0,0.762884,0.035613,0,0,0.195357,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p26.pdf", p26];

rsys=DISEASE[0.874330,0,0.070383,0,0,0.119677,0,0.832078,0.095579,0,0.439693,0,0.615957,0,0,0.490706,0,0.147019,0.692261,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["27: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p27.pdf", p27];

rsys=DISEASE[0,0.721010,0,0.828175,0,0.933016,0,0.644192,0,0.925505,0,1.043142,0,0.473604,0,1.550701,0,0.217169,0,1.632497];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p28.pdf", p28];

rsys=DISEASE[0.299678,0,0.532148,0,0.943522,0,1.728087,0,1.646257,0,1.563219,0,0.792722,0,2.086159,0,1.955717,0,0.567490,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["29: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p29.pdf", p29];

rsys=DISEASE[1.901986,0,2.149174,0,1.217220,0,1.011548,0,0.656816,0,0,0.676668,1.135337,0,1.061452,0,0,0.203039,1.250057,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["30: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p30.pdf", p30];

rsys=DISEASE[0.745714,0,1.452460,0,1.017044,0,2.787653,0,1.047604,0,0,0.028091,0,0.457603,1.380942,0,1.722711,0,0.884920,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["31: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p31.pdf", p31];

rsys=DISEASE[0.670741,0,0,0.479724,0,0.564256,0,0.997606,0,2.417518,0,1.649045,0.064826,0,0,0.020246,0,1.395325,0.325290,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p32.pdf", p32];

rsys=DISEASE[2.509373,0,1.761186,0,0.541448,0,0.398436,0,0,0.492681,0.313300,0,1.211277,0,0,0.670359,0,2.273552,0,0.845345];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["33: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p33.pdf", p33];

rsys=DISEASE[0,1.184787,0,0.359601,1.442381,0,0,0.004885,0.050310,0,0.145862,0,0,0.076131,0.115814,0,0.655109,0,0.145475,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["34: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p34.pdf", p34];

rsys=DISEASE[0,0.080889,0,0.405001,0,1.500515,0,0.283552,0,0.775148,0,0.522912,0,0.637673,0,0.337094,0,1.762678,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p35.pdf", p35];

rsys=DISEASE[0,1.260815,0,1.437493,0,1.350886,0,1.436157,0,1.165474,0,1.082232,0,1.086957,0,0.165554,0,0.737958,0,0.465531];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p36.pdf", p36];

rsys=DISEASE[0,1.113404,0,1.209165,0,1.185461,0,0.226870,0,0.765679,0,0.360361,0,0.953366,0,1.035896,0,0.040186,0,1.203141];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p37.pdf", p37];

rsys=DISEASE[0,0.164521,0,0.218102,0,0.479247,0,2.090821,0,2.485883,0,1.037930,0,0.035494,0,0.579400,0,1.468697,0,1.124244];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p38.pdf", p38];

rsys=DISEASE[0,1.046879,0,0.215643,0.589122,0,0.410179,0,0.559813,0,1.518916,0,0.185720,0,0.314528,0,1.245551,0,0,0.184799];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["39: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p39.pdf", p39];

rsys=DISEASE[0,0.018377,0,0.498452,0,0.053622,0,0.886050,0,2.599978,0,1.134353,0,0.311313,0,0.225376,0,0.880983,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p40.pdf", p40];

rsys=DISEASE[0,1.349305,0,1.102851,0,1.005677,0,1.133327,0,1.123901,0,0.552555,0,0.391315,0,1.411809,0,0.489586,0,1.417820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p41.pdf", p41];

rsys=DISEASE[0,0.302007,0,0.432242,0,0.074013,0,0.683485,0.273650,0,1.001293,0,0.295182,0,0,0.114225,0.489279,0,0,0.096726];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["42: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p42.pdf", p42];

rsys=DISEASE[0,0.228723,0,0.999942,0,1.406028,0,0.556798,0,0.718562,0,0.642790,0,0.471826,0,0.475043,0,0.169330,0,1.755433];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p43.pdf", p43];

rsys=DISEASE[0,0.394086,0,0.778802,0,0.961449,0,0.283552,0,0.957609,0,0.490662,0,0.762884,0,0.914931,0,0.831903,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p44.pdf", p44];

rsys=DISEASE[0.106860,0,0,0.049740,0,0.961449,0,1.399122,0,3.785969,0,1.456198,0.190290,0,0,0.765847,0,1.248335,0,0.581127];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p45.pdf", p45];

rsys=DISEASE[0,0.331573,0,0.685352,0,0.848006,0,0.372752,0,0.364499,0,0.812508,0,0.461921,0,0.299918,0,1.199256,0,1.305894];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p46.pdf", p46];

rsys=DISEASE[0,1.367678,0,0.697081,0,1.166219,0,0.864145,0,0.228692,0,0.431049,0,0.387252,0,0.930593,0,0.315823,0,1.544425];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p47.pdf", p47];

rsys=DISEASE[1.005479,0,0.695591,0,0.833239,0,0.627874,0,1.510681,0,1.570060,0,2.062605,0,1.003140,0,0.110770,0,1.525284,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["48: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p48.pdf", p48];

rsys=DISEASE[0.886578,0,0.761800,0,1.359956,0,0.392565,0,1.228446,0,2.276622,0,1.441887,0,0.623450,0,0.946362,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["49: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p49.pdf", p49];

rsys=DISEASE[0.044347,0,0.606113,0,1.334109,0,1.735314,0,1.027740,0,1.287305,0,1.201117,0,1.256580,0,1.526890,0,0.631711,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p50.pdf", p50];

rsys=DISEASE[0.420479,0,0.534607,0,1.167248,0,1.225855,0,1.335152,0,2.072373,0,1.592750,0,1.111838,0,1.526890,0,1.703265,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p51.pdf", p51];

rsys=DISEASE[0.446456,0,0.276767,0,0.943522,0,1.400191,0,1.795228,0,1.074914,0,1.290517,0,1.861024,0,1.551678,0,1.361982,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["52: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p52.pdf", p52];

rsys=DISEASE[0.065255,0,0,0.834986,0,0.989881,0,0.952894,0,0.775148,0,0.651585,0,0.813170,0,0.803024,0,1.150424,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p53.pdf", p53];

rsys=DISEASE[0.998088,0,1.226591,0,1.360244,0,1.101652,0,0.541797,0,0,0.709243,0.275626,0,1.700242,0,0.216615,0,0.884920,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p54.pdf", p54];

rsys=DISEASE[1.586678,0,2.544729,0,0.673845,0,1.978301,0,1.001642,0,0,0.903718,1.084797,0,1.174302,0,0,0.128925,1.068406,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p55.pdf", p55];

rsys=DISEASE[0.065255,0,0,0.909708,0,0.621121,0,0.774493,0,0.752282,0,1.005680,0,0.788028,0,1.026649,0,1.346245,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p56.pdf", p56];

rsys=DISEASE[0.190702,0,0.570360,0,1.445540,0,1.314604,0,1.203272,0,1.350176,0,1.174958,0,1.600791,0,1.526890,0,0.453730,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["57: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p57.pdf", p57];

rsys=DISEASE[1.138107,0,1.068067,0,1.263746,0,0.931607,0,0.333700,0,1.487970,0,2.610433,0,1.478505,0,0.466474,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p58.pdf", p58];

rsys=DISEASE[0,0.080889,0,0.629358,0,0.848006,0,0.752136,0,0.843513,0,0.201066,0,0.788028,0,0.318411,0,0.880983,0,0.399476];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p59.pdf", p59];

rsys=DISEASE[0,0.540231,0,0.517180,0,0.422382,0,0.618223,0.091651,0,0,1.584871,0,0.512209,0,0.579400,0,1.762678,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p60.pdf", p60];

rsys=DISEASE[0.810338,0,0.051088,0,0,0.148397,0.567353,0,0.858445,0,0,0.670153,0,0.558178,0,0.528636,0.929507,0,0.323456,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["61: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p61.pdf", p61];

rsys=DISEASE[0,1.307489,0,1.084501,0,0.678848,0,0.671291,0,0.815337,0,0.797849,0,0.391315,0,1.220833,0,0.824714,0,1.456352];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p62.pdf", p62];

rsys=DISEASE[1.263977,0,1.374333,0,1.039446,0,0.509768,0,0.810868,0,1.375259,0,1.788057,0,0.556080,0,0.466474,0,0.965653,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["63: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p63.pdf", p63];

rsys=DISEASE[0,1.244554,0,1.231108,0,1.259845,0,0.209255,0,0.197974,0,0.347982,0,1.336616,0,0.439565,0.117711,0,0,0.929750];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p64.pdf", p64];

rsys=DISEASE[0,1.253212,0,0.983485,0,0.588094,1.468390,0,0.123755,0,1.687657,0,0,0.344075,0,0.344831,1.559112,0,0,0.755438];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["65: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p65.pdf", p65];

rsys=DISEASE[0,0.298416,0,0.055604,0.276079,0,0,0.616416,0.351484,0,1.429334,0,0.656593,0,0,0.119131,0.371786,0,0,0.019662];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["66: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p66.pdf", p66];

rsys=DISEASE[0.276024,0,0.031982,0,0.515887,0,0,0.899825,0.026059,0,0.766099,0,0,0.405030,0,1.040613,0.654116,0,1.244551,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["67: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p67.pdf", p67];

rsys=DISEASE[0,0.644559,0,0.760074,0,0.734563,0,1.287565,0,0.912109,0,0.587412,0,1.339918,0,1.287637,0,1.125884,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p68.pdf", p68];

rsys=DISEASE[0,1.579714,0,1.304317,0,0.642661,0.415825,0,0.243163,0,0,0.675365,0,1.953017,0,0.352757,0.243385,0,0,0.656356];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p69.pdf", p69];

rsys=DISEASE[0.232307,0,0,0.498452,0,0.479247,0,0.283552,0.319612,0,0.152703,0,0,0.487065,0.035613,0,0,1.321707,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p70.pdf", p70];

rsys=DISEASE[0.572114,0,0.940566,0,0.655177,0,1.493908,0,1.407442,0,1.901027,0,1.052542,0,1.298663,0,1.628520,0,1.760145,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["71: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p71.pdf", p71];

rsys=DISEASE[0,1.387319,0,0.619332,0,0.992466,0,0.402786,0,0.718562,0,0.956817,0,1.033622,0,0.958523,0,0.453642,0,1.386627];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p72.pdf", p72];

rsys=DISEASE[0.023439,0,0,0.647897,0.400147,0,0,0.171994,0.068786,0,0,1.295276,0,0.210993,0.240742,0,0,1.493237,0,1.124244];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p73.pdf", p73];

rsys=DISEASE[0,0.227034,0,0.797530,0,1.018026,0,0.774493,0,2.189329,0,1.295276,0,0.361601,0,0.243870,0,1.027972,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p74.pdf", p74];

rsys=DISEASE[0.886578,0,1.119143,0,1.135657,0,0.111414,0,0.781073,0,1.826104,0,1.788057,0,1.208647,0,1.148135,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p75.pdf", p75];

rsys=DISEASE[0,0.059981,0,1.152603,0,1.103323,0,0.752136,0,0.569823,0,1.906717,0,0.963779,0,0.821706,0,1.468697,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p76.pdf", p76];

rsys=DISEASE[0.796610,0,0.695591,0,1.000100,0,1.137332,0,1.708156,0,0.973279,0,1.514271,0,2.342618,0,1.184325,0,1.525284,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["77: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p77.pdf", p77];

rsys=DISEASE[0,0.416683,0,0.359601,0.304798,0,0,0.310651,0,0.111825,0.653387,0,0.133400,0,0.097699,0,0.678904,0,0,0.349936];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["78: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p78.pdf", p78];

rsys=DISEASE[0,0.310665,0,0.479724,0,0.479247,0,1.711662,0,0.638420,0,0.877007,0,1.139277,0,0.448999,0,0.611541,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p79.pdf", p79];

rsys=DISEASE[1.237789,0,0.319710,0,0,0.090671,0,0.087310,0,0.574905,0,0.147969,0,0.430682,0,0.452775,0,0.447447,0.139970,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["80: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p80.pdf", p80];

rsys=DISEASE[0.802735,0,0.685186,0,1.135657,0,1.728087,0,1.735640,0,1.600681,0,1.701452,0,1.433592,0,1.072532,0,1.561982,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p81.pdf", p81];

rsys=DISEASE[1.259120,0,0.242906,0,0,0.292857,0.251426,0,0,0.389904,0,0.213119,0.615957,0,0,0.263308,0,0.547836,0.508776,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["82: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p82.pdf", p82];

rsys=DISEASE[0.504111,0,0.480882,0,0.805381,0,1.668922,0,1.510681,0,1.130292,0,1.044413,0,1.637024,0,1.769807,0,1.345469,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["83: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p83.pdf", p83];

rsys=DISEASE[0,0.253222,0.276578,0,1.006706,0,0,0.165897,0.690768,0,1.405228,0,0.309153,0,0.206775,0,0.852913,0,0,0.806814];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["84: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p84.pdf", p84];

rsys=DISEASE[0.655957,0,0.838414,0,1.231867,0,1.048583,0,1.258472,0,1.713392,0,1.485063,0,0.871042,0,0.845230,0,0.765656,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p85.pdf", p85];

rsys=DISEASE[0.461873,0,0,0.012473,0,0.337372,0,1.354634,0,3.261454,0,0.522912,0.165147,0,0.557590,0,0,1.297167,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p86.pdf", p86];

rsys=DISEASE[0,0.821960,0,0.411055,0.116110,0,0,0.478664,0.102046,0,0.421450,0,0,0.512716,0,0.617709,0.778303,0,0,0.096726];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["87: 3 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p87.pdf", p87];

rsys=DISEASE[1.410543,0,1.399871,0,1.039446,0,0.790918,0,0.840661,0,1.938814,0,1.961269,0,1.951041,0,0.744345,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p88.pdf", p88];

rsys=DISEASE[0,1.513611,0,1.190247,0,1.323316,0,1.436157,0,0.412770,0,1.521675,0,1.220803,0,0.842464,0,0.427864,0,0.834337];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p89.pdf", p89];

rsys=DISEASE[0,1.485945,0,1.175492,0,1.761289,0.675974,0,0.032295,0,0,0.466230,0,1.164421,0,1.245553,1.849129,0,0,2.128092];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["90: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p90.pdf", p90];

rsys=DISEASE[0.489749,0,0.223800,0,0.515887,0,0.318946,0,0.581061,0,0.439693,0,0,0.762375,0,0.073652,0.278584,0,0.323456,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["91: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p91.pdf", p91];

rsys=DISEASE[1.187315,0,2.393961,0,0.931172,0,1.596207,0,0.495835,0,0.782712,0,0.124002,0,0.892365,0,0.488040,0,1.250057,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p92.pdf", p92];

rsys=DISEASE[0,1.429346,0,0.657544,0,1.433599,0,0.952442,0,0.883239,0,0.736932,0,0.391569,0,1.383880,0,0.892880,0,0.834337];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p93.pdf", p93];

rsys=DISEASE[0,1.336210,0,0.823257,0,1.690352,0.078444,0,0,0.195666,0,0.132332,0,0.021525,0,0.905873,0.997177,0,0,0.087551];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["94: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p94.pdf", p94];

rsys=DISEASE[0.253004,0,0,0.909708,0,0.621121,0,0.997606,0,0.387362,0,1.230777,0,0.411887,0,0.150647,0,1.591148,0,0.581127];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p95.pdf", p95];

rsys=DISEASE[0,0.695879,0,1.074854,0.021910,0,0,0.346782,0,0.525709,0,1.237291,0,1.598467,0,0.621671,1.673137,0,0,1.254519];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["96: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p96.pdf", p96];

rsys=DISEASE[0,0.310665,0,0.386274,0,1.018026,0,1.265208,0,2.850803,0,0.973431,0,0.637673,0.166201,0,0,0.538170,0,0.399476];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p97.pdf", p97];

rsys=DISEASE[0,0.215418,0.185399,0,0.544607,0,1.696021,0,0.742734,0,0.341640,0,0.054414,0,0.286600,0,0.579009,0,0,0.412320];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["98: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p98.pdf", p98];

rsys=DISEASE[0,0.059981,0,0.610630,0,1.046459,0,1.845350,0,2.326289,0,1.263026,0,0.336457,0.277919,0,0,0.807611,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASE_p99.pdf", p99];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
tmax=50;
totalCount=0;
correctCount=0;
matchingCount=0;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=DISEASEReduced[0.652789,0,0,0.023067,0.461607,0,0,0.272034,0,0.438868,0,0.362966,0,0.623959,0,0.579211,0,0.149747,0,0.474706];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["0: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p0.pdf", p0];

rsys=DISEASEReduced[0.880243,0,1.053501,0,1.529114,0,1.159462,0,1.181329,0,1.381449,0,1.932063,0,2.162209,0,0.964706,0,1.525284,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p1.pdf", p1];

rsys=DISEASEReduced[1.061022,0,1.659602,0,0.588261,0,1.641145,0,1.047604,0,0,0.255142,0.301024,0,1.625134,0,0.265941,0,1.615191,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["2: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p2.pdf", p2];

rsys=DISEASEReduced[0,1.197669,0,1.057072,0,0.964895,0,1.524228,0,1.636175,0,0.799803,0,0.445159,0,0.900400,0,0.944686,0,0.834337];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p3.pdf", p3];

rsys=DISEASEReduced[0,0.762404,0,1.292778,0,1.832227,0,0.462179,0,0.325927,0.747530,0,0,1.192358,0,0.978716,1.050472,0,0,1.316903];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["4: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p4.pdf", p4];

rsys=DISEASEReduced[0,0.867577,0,0.846525,0,1.041863,0,1.432318,0,1.212129,0,1.002422,0,0.411887,0,1.550701,0,1.033922,0,1.867359];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p5.pdf", p5];

rsys=DISEASEReduced[0,1.597877,0,1.133118,0,0.937324,0,1.216430,0,0.506771,0,1.176376,0,1.434652,0,0.494291,0,0.298472,0,0.465531];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p6.pdf", p6];

rsys=DISEASEReduced[0,0.289758,0,0.460996,0,0.933016,0,1.131520,0,0.364499,0,0.716085,0,0.562242,0,0.411636,0,1.713598,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p7.pdf", p7];

rsys=DISEASEReduced[1.216459,0,0.146997,0,0,0.206410,0.251426,0,0,0.505615,0.178437,0,0.769104,0,0,0.623370,0,0.898334,0,0.228836];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["8: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p8.pdf", p8];

rsys=DISEASEReduced[0,0.227034,0,0.704080,0,0.252074,0,0.796849,0,1.025974,0,1.745794,0,1.089244,0,0.411636,0,0.856443,0,0.762777];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["9: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p9.pdf", p9];

rsys=DISEASEReduced[0.553952,0,0.300414,0,0.515887,0,0,0.064729,0.650349,0,0,0.049916,0.182164,0,0,0.073652,0.053265,0,0.323456,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["10: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p10.pdf", p10];

rsys=DISEASEReduced[0.361556,0,0,0.255746,0.977986,0,0,0.109893,0,0.043231,0,0.115067,0,1.374966,0,1.078543,1.455253,0,0.139970,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["11: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p11.pdf", p11];

rsys=DISEASEReduced[0,1.436315,0,0.786748,0,0.823595,0,0.806108,0,0.450186,0,0.908605,0,0.858379,0,1.164785,0,0.932044,0,1.968276];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p12.pdf", p12];

rsys=DISEASEReduced[1.523531,0,1.979867,0,0.445238,0,0.516994,0,0,0.148087,0,0.319967,0.756151,0,1.136748,0,0.068385,0,1.433541,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["13: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p13.pdf", p13];

rsys=DISEASEReduced[0.959861,0,0.703346,0,0.342421,0,0,0.087310,0.812022,0,0.766099,0,0.513858,0,0,0.358041,0.378727,0,0,0.228836];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["14: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p14.pdf", p14];

rsys=DISEASEReduced[0.880243,0,0.695591,0,1.167248,0,2.156249,0,1.554564,0,0.973279,0,1.357822,0,2.162209,0,1.184325,0,2.061060,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p15.pdf", p15];

rsys=DISEASEReduced[0.127345,0,0.370217,0,0.849323,0,0.468893,0,0.371810,0,1.103580,0,1.518587,0,0.405678,0,0,0.890649,0,0.476540];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["16: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p16.pdf", p16];

rsys=DISEASEReduced[0.746136,0,0.127890,0,0,0.408596,0,0.606480,0.511540,0,0.276490,0,0.233213,0,0.608919,0,0.253547,0,0.875746,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["17: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p17.pdf", p17];

rsys=DISEASEReduced[1.590691,0,3.880838,0,2.029985,0,2.975990,0,1.817862,0,1.821217,0,3.315473,0,0.260557,0,0,0.670039,0,0.082048];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["18: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p18.pdf", p18];

rsys=DISEASEReduced[0,1.471585,0,0.372086,0,0.330479,0,2.997277,0,1.400824,0.079082,0,0,0.471826,0,2.486336,0,0.763736,0,2.124788];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p19.pdf", p19];

rsys=DISEASEReduced[0,0.080889,0,0.666625,0,0.961449,0.051119,0,0,0.387362,0,1.327525,0,0.186103,0,0.411636,0,1.615688,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p20.pdf", p20];

rsys=DISEASEReduced[1.712759,0,2.055157,0,1.045764,0,1.146591,0,0.403913,0,0.231210,0,0.225086,0,0.948791,0,0,0.894119,1.615191,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["21: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p21.pdf", p21];

rsys=DISEASEReduced[0,0.613514,0,0.231533,0.247646,0,0.563061,0,1.046218,0,0.414609,0,0.656593,0,0,0.191409,0.560667,0,0.145475,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["22: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p22.pdf", p22];

rsys=DISEASEReduced[0.331778,0,0.008335,0,0.390096,0,0,0.660000,0,0.227538,1.011392,0,0.447315,0,0,0.010811,0.112011,0,0,0.184799];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["23: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p23.pdf", p23];

rsys=DISEASEReduced[0,0.787325,0,1.175492,0,0.341680,0.562159,0,0.357720,0,0.504844,0,0,0.328075,0,0.663378,0,0.147515,0,0.087551];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["24: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p24.pdf", p24];

rsys=DISEASEReduced[0,1.197669,0,1.285400,0,0.992466,0,1.150264,0,0.318537,0,0.705660,0,0.525415,0,1.519376,0,0.401837,0,0.280212];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["25: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p25.pdf", p25];

rsys=DISEASEReduced[0,0.289758,0,0.685352,0,0.677699,0,0.908407,0,3.192858,0,1.359450,0,0.762884,0.035613,0,0,0.195357,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p26.pdf", p26];

rsys=DISEASEReduced[0.874330,0,0.070383,0,0,0.119677,0,0.832078,0.095579,0,0.439693,0,0.615957,0,0,0.490706,0,0.147019,0.692261,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["27: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p27.pdf", p27];

rsys=DISEASEReduced[0,0.721010,0,0.828175,0,0.933016,0,0.644192,0,0.925505,0,1.043142,0,0.473604,0,1.550701,0,0.217169,0,1.632497];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p28.pdf", p28];

rsys=DISEASEReduced[0.299678,0,0.532148,0,0.943522,0,1.728087,0,1.646257,0,1.563219,0,0.792722,0,2.086159,0,1.955717,0,0.567490,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["29: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p29.pdf", p29];

rsys=DISEASEReduced[1.901986,0,2.149174,0,1.217220,0,1.011548,0,0.656816,0,0,0.676668,1.135337,0,1.061452,0,0,0.203039,1.250057,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["30: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p30.pdf", p30];

rsys=DISEASEReduced[0.745714,0,1.452460,0,1.017044,0,2.787653,0,1.047604,0,0,0.028091,0,0.457603,1.380942,0,1.722711,0,0.884920,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["31: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p31.pdf", p31];

rsys=DISEASEReduced[0.670741,0,0,0.479724,0,0.564256,0,0.997606,0,2.417518,0,1.649045,0.064826,0,0,0.020246,0,1.395325,0.325290,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p32.pdf", p32];

rsys=DISEASEReduced[2.509373,0,1.761186,0,0.541448,0,0.398436,0,0,0.492681,0.313300,0,1.211277,0,0,0.670359,0,2.273552,0,0.845345];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["33: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p33.pdf", p33];

rsys=DISEASEReduced[0,1.184787,0,0.359601,1.442381,0,0,0.004885,0.050310,0,0.145862,0,0,0.076131,0.115814,0,0.655109,0,0.145475,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["34: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p34.pdf", p34];

rsys=DISEASEReduced[0,0.080889,0,0.405001,0,1.500515,0,0.283552,0,0.775148,0,0.522912,0,0.637673,0,0.337094,0,1.762678,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p35.pdf", p35];

rsys=DISEASEReduced[0,1.260815,0,1.437493,0,1.350886,0,1.436157,0,1.165474,0,1.082232,0,1.086957,0,0.165554,0,0.737958,0,0.465531];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p36.pdf", p36];

rsys=DISEASEReduced[0,1.113404,0,1.209165,0,1.185461,0,0.226870,0,0.765679,0,0.360361,0,0.953366,0,1.035896,0,0.040186,0,1.203141];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p37.pdf", p37];

rsys=DISEASEReduced[0,0.164521,0,0.218102,0,0.479247,0,2.090821,0,2.485883,0,1.037930,0,0.035494,0,0.579400,0,1.468697,0,1.124244];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p38.pdf", p38];

rsys=DISEASEReduced[0,1.046879,0,0.215643,0.589122,0,0.410179,0,0.559813,0,1.518916,0,0.185720,0,0.314528,0,1.245551,0,0,0.184799];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["39: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p39.pdf", p39];

rsys=DISEASEReduced[0,0.018377,0,0.498452,0,0.053622,0,0.886050,0,2.599978,0,1.134353,0,0.311313,0,0.225376,0,0.880983,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p40.pdf", p40];

rsys=DISEASEReduced[0,1.349305,0,1.102851,0,1.005677,0,1.133327,0,1.123901,0,0.552555,0,0.391315,0,1.411809,0,0.489586,0,1.417820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p41.pdf", p41];

rsys=DISEASEReduced[0,0.302007,0,0.432242,0,0.074013,0,0.683485,0.273650,0,1.001293,0,0.295182,0,0,0.114225,0.489279,0,0,0.096726];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["42: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p42.pdf", p42];

rsys=DISEASEReduced[0,0.228723,0,0.999942,0,1.406028,0,0.556798,0,0.718562,0,0.642790,0,0.471826,0,0.475043,0,0.169330,0,1.755433];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p43.pdf", p43];

rsys=DISEASEReduced[0,0.394086,0,0.778802,0,0.961449,0,0.283552,0,0.957609,0,0.490662,0,0.762884,0,0.914931,0,0.831903,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p44.pdf", p44];

rsys=DISEASEReduced[0.106860,0,0,0.049740,0,0.961449,0,1.399122,0,3.785969,0,1.456198,0.190290,0,0,0.765847,0,1.248335,0,0.581127];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p45.pdf", p45];

rsys=DISEASEReduced[0,0.331573,0,0.685352,0,0.848006,0,0.372752,0,0.364499,0,0.812508,0,0.461921,0,0.299918,0,1.199256,0,1.305894];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p46.pdf", p46];

rsys=DISEASEReduced[0,1.367678,0,0.697081,0,1.166219,0,0.864145,0,0.228692,0,0.431049,0,0.387252,0,0.930593,0,0.315823,0,1.544425];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p47.pdf", p47];

rsys=DISEASEReduced[1.005479,0,0.695591,0,0.833239,0,0.627874,0,1.510681,0,1.570060,0,2.062605,0,1.003140,0,0.110770,0,1.525284,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["48: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p48.pdf", p48];

rsys=DISEASEReduced[0.886578,0,0.761800,0,1.359956,0,0.392565,0,1.228446,0,2.276622,0,1.441887,0,0.623450,0,0.946362,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["49: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p49.pdf", p49];

rsys=DISEASEReduced[0.044347,0,0.606113,0,1.334109,0,1.735314,0,1.027740,0,1.287305,0,1.201117,0,1.256580,0,1.526890,0,0.631711,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p50.pdf", p50];

rsys=DISEASEReduced[0.420479,0,0.534607,0,1.167248,0,1.225855,0,1.335152,0,2.072373,0,1.592750,0,1.111838,0,1.526890,0,1.703265,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p51.pdf", p51];

rsys=DISEASEReduced[0.446456,0,0.276767,0,0.943522,0,1.400191,0,1.795228,0,1.074914,0,1.290517,0,1.861024,0,1.551678,0,1.361982,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["52: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p52.pdf", p52];

rsys=DISEASEReduced[0.065255,0,0,0.834986,0,0.989881,0,0.952894,0,0.775148,0,0.651585,0,0.813170,0,0.803024,0,1.150424,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p53.pdf", p53];

rsys=DISEASEReduced[0.998088,0,1.226591,0,1.360244,0,1.101652,0,0.541797,0,0,0.709243,0.275626,0,1.700242,0,0.216615,0,0.884920,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p54.pdf", p54];

rsys=DISEASEReduced[1.586678,0,2.544729,0,0.673845,0,1.978301,0,1.001642,0,0,0.903718,1.084797,0,1.174302,0,0,0.128925,1.068406,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p55.pdf", p55];

rsys=DISEASEReduced[0.065255,0,0,0.909708,0,0.621121,0,0.774493,0,0.752282,0,1.005680,0,0.788028,0,1.026649,0,1.346245,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p56.pdf", p56];

rsys=DISEASEReduced[0.190702,0,0.570360,0,1.445540,0,1.314604,0,1.203272,0,1.350176,0,1.174958,0,1.600791,0,1.526890,0,0.453730,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["57: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p57.pdf", p57];

rsys=DISEASEReduced[1.138107,0,1.068067,0,1.263746,0,0.931607,0,0.333700,0,1.487970,0,2.610433,0,1.478505,0,0.466474,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p58.pdf", p58];

rsys=DISEASEReduced[0,0.080889,0,0.629358,0,0.848006,0,0.752136,0,0.843513,0,0.201066,0,0.788028,0,0.318411,0,0.880983,0,0.399476];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p59.pdf", p59];

rsys=DISEASEReduced[0,0.540231,0,0.517180,0,0.422382,0,0.618223,0.091651,0,0,1.584871,0,0.512209,0,0.579400,0,1.762678,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p60.pdf", p60];

rsys=DISEASEReduced[0.810338,0,0.051088,0,0,0.148397,0.567353,0,0.858445,0,0,0.670153,0,0.558178,0,0.528636,0.929507,0,0.323456,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["61: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p61.pdf", p61];

rsys=DISEASEReduced[0,1.307489,0,1.084501,0,0.678848,0,0.671291,0,0.815337,0,0.797849,0,0.391315,0,1.220833,0,0.824714,0,1.456352];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p62.pdf", p62];

rsys=DISEASEReduced[1.263977,0,1.374333,0,1.039446,0,0.509768,0,0.810868,0,1.375259,0,1.788057,0,0.556080,0,0.466474,0,0.965653,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["63: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p63.pdf", p63];

rsys=DISEASEReduced[0,1.244554,0,1.231108,0,1.259845,0,0.209255,0,0.197974,0,0.347982,0,1.336616,0,0.439565,0.117711,0,0,0.929750];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p64.pdf", p64];

rsys=DISEASEReduced[0,1.253212,0,0.983485,0,0.588094,1.468390,0,0.123755,0,1.687657,0,0,0.344075,0,0.344831,1.559112,0,0,0.755438];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["65: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p65.pdf", p65];

rsys=DISEASEReduced[0,0.298416,0,0.055604,0.276079,0,0,0.616416,0.351484,0,1.429334,0,0.656593,0,0,0.119131,0.371786,0,0,0.019662];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["66: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p66.pdf", p66];

rsys=DISEASEReduced[0.276024,0,0.031982,0,0.515887,0,0,0.899825,0.026059,0,0.766099,0,0,0.405030,0,1.040613,0.654116,0,1.244551,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["67: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p67.pdf", p67];

rsys=DISEASEReduced[0,0.644559,0,0.760074,0,0.734563,0,1.287565,0,0.912109,0,0.587412,0,1.339918,0,1.287637,0,1.125884,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p68.pdf", p68];

rsys=DISEASEReduced[0,1.579714,0,1.304317,0,0.642661,0.415825,0,0.243163,0,0,0.675365,0,1.953017,0,0.352757,0.243385,0,0,0.656356];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p69.pdf", p69];

rsys=DISEASEReduced[0.232307,0,0,0.498452,0,0.479247,0,0.283552,0.319612,0,0.152703,0,0,0.487065,0.035613,0,0,1.321707,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p70.pdf", p70];

rsys=DISEASEReduced[0.572114,0,0.940566,0,0.655177,0,1.493908,0,1.407442,0,1.901027,0,1.052542,0,1.298663,0,1.628520,0,1.760145,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["71: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p71.pdf", p71];

rsys=DISEASEReduced[0,1.387319,0,0.619332,0,0.992466,0,0.402786,0,0.718562,0,0.956817,0,1.033622,0,0.958523,0,0.453642,0,1.386627];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p72.pdf", p72];

rsys=DISEASEReduced[0.023439,0,0,0.647897,0.400147,0,0,0.171994,0.068786,0,0,1.295276,0,0.210993,0.240742,0,0,1.493237,0,1.124244];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p73.pdf", p73];

rsys=DISEASEReduced[0,0.227034,0,0.797530,0,1.018026,0,0.774493,0,2.189329,0,1.295276,0,0.361601,0,0.243870,0,1.027972,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p74.pdf", p74];

rsys=DISEASEReduced[0.886578,0,1.119143,0,1.135657,0,0.111414,0,0.781073,0,1.826104,0,1.788057,0,1.208647,0,1.148135,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p75.pdf", p75];

rsys=DISEASEReduced[0,0.059981,0,1.152603,0,1.103323,0,0.752136,0,0.569823,0,1.906717,0,0.963779,0,0.821706,0,1.468697,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p76.pdf", p76];

rsys=DISEASEReduced[0.796610,0,0.695591,0,1.000100,0,1.137332,0,1.708156,0,0.973279,0,1.514271,0,2.342618,0,1.184325,0,1.525284,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["77: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p77.pdf", p77];

rsys=DISEASEReduced[0,0.416683,0,0.359601,0.304798,0,0,0.310651,0,0.111825,0.653387,0,0.133400,0,0.097699,0,0.678904,0,0,0.349936];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["78: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p78.pdf", p78];

rsys=DISEASEReduced[0,0.310665,0,0.479724,0,0.479247,0,1.711662,0,0.638420,0,0.877007,0,1.139277,0,0.448999,0,0.611541,0,0.217826];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p79.pdf", p79];

rsys=DISEASEReduced[1.237789,0,0.319710,0,0,0.090671,0,0.087310,0,0.574905,0,0.147969,0,0.430682,0,0.452775,0,0.447447,0.139970,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["80: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p80.pdf", p80];

rsys=DISEASEReduced[0.802735,0,0.685186,0,1.135657,0,1.728087,0,1.735640,0,1.600681,0,1.701452,0,1.433592,0,1.072532,0,1.561982,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p81.pdf", p81];

rsys=DISEASEReduced[1.259120,0,0.242906,0,0,0.292857,0.251426,0,0,0.389904,0,0.213119,0.615957,0,0,0.263308,0,0.547836,0.508776,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["82: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p82.pdf", p82];

rsys=DISEASEReduced[0.504111,0,0.480882,0,0.805381,0,1.668922,0,1.510681,0,1.130292,0,1.044413,0,1.637024,0,1.769807,0,1.345469,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["83: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p83.pdf", p83];

rsys=DISEASEReduced[0,0.253222,0.276578,0,1.006706,0,0,0.165897,0.690768,0,1.405228,0,0.309153,0,0.206775,0,0.852913,0,0,0.806814];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 3, matchingCount+=1, warning = "**"];
Print["84: 3 " <> ToString[predicted] <> " 3 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p84.pdf", p84];

rsys=DISEASEReduced[0.655957,0,0.838414,0,1.231867,0,1.048583,0,1.258472,0,1.713392,0,1.485063,0,0.871042,0,0.845230,0,0.765656,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p85.pdf", p85];

rsys=DISEASEReduced[0.461873,0,0,0.012473,0,0.337372,0,1.354634,0,3.261454,0,0.522912,0.165147,0,0.557590,0,0,1.297167,0,0.038011];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p86.pdf", p86];

rsys=DISEASEReduced[0,0.821960,0,0.411055,0.116110,0,0,0.478664,0.102046,0,0.421450,0,0,0.512716,0,0.617709,0.778303,0,0,0.096726];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 3, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["87: 3 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p87.pdf", p87];

rsys=DISEASEReduced[1.410543,0,1.399871,0,1.039446,0,0.790918,0,0.840661,0,1.938814,0,1.961269,0,1.951041,0,0.744345,0,1.163818,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p88.pdf", p88];

rsys=DISEASEReduced[0,1.513611,0,1.190247,0,1.323316,0,1.436157,0,0.412770,0,1.521675,0,1.220803,0,0.842464,0,0.427864,0,0.834337];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p89.pdf", p89];

rsys=DISEASEReduced[0,1.485945,0,1.175492,0,1.761289,0.675974,0,0.032295,0,0,0.466230,0,1.164421,0,1.245553,1.849129,0,0,2.128092];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["90: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p90.pdf", p90];

rsys=DISEASEReduced[0.489749,0,0.223800,0,0.515887,0,0.318946,0,0.581061,0,0.439693,0,0,0.762375,0,0.073652,0.278584,0,0.323456,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["91: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p91.pdf", p91];

rsys=DISEASEReduced[1.187315,0,2.393961,0,0.931172,0,1.596207,0,0.495835,0,0.782712,0,0.124002,0,0.892365,0,0.488040,0,1.250057,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p92.pdf", p92];

rsys=DISEASEReduced[0,1.429346,0,0.657544,0,1.433599,0,0.952442,0,0.883239,0,0.736932,0,0.391569,0,1.383880,0,0.892880,0,0.834337];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p93.pdf", p93];

rsys=DISEASEReduced[0,1.336210,0,0.823257,0,1.690352,0.078444,0,0,0.195666,0,0.132332,0,0.021525,0,0.905873,0.997177,0,0,0.087551];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["94: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p94.pdf", p94];

rsys=DISEASEReduced[0.253004,0,0,0.909708,0,0.621121,0,0.997606,0,0.387362,0,1.230777,0,0.411887,0,0.150647,0,1.591148,0,0.581127];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p95.pdf", p95];

rsys=DISEASEReduced[0,0.695879,0,1.074854,0.021910,0,0,0.346782,0,0.525709,0,1.237291,0,1.598467,0,0.621671,1.673137,0,0,1.254519];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["96: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p96.pdf", p96];

rsys=DISEASEReduced[0,0.310665,0,0.386274,0,1.018026,0,1.265208,0,2.850803,0,0.973431,0,0.637673,0.166201,0,0,0.538170,0,0.399476];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p97.pdf", p97];

rsys=DISEASEReduced[0,0.215418,0.185399,0,0.544607,0,1.696021,0,0.742734,0,0.341640,0,0.054414,0,0.286600,0,0.579009,0,0,0.412320];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 2, correctCount+=1, warning = "*"];
If[predicted == 2, matchingCount+=1, warning = "**"];
Print["98: 2 " <> ToString[predicted] <> " 2 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p98.pdf", p98];

rsys=DISEASEReduced[0,0.059981,0,0.610630,0,1.046459,0,1.845350,0,2.326289,0,1.263026,0,0.336457,0.277919,0,0,0.807611,0.143639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/DISEASEReduced_p99.pdf", p99];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
