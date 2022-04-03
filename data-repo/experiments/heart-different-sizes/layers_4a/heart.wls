#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

HEART[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x0Sneg, x0SnegInitial],
conc[x1Spos, x1SposInitial],
conc[x1Sneg, x1SnegInitial],
rxn[x0Spos,iL0N0Spos+iL0N2Sneg+iL0N4Sneg,1],
rxn[x0Sneg,iL0N0Sneg+iL0N2Spos+iL0N4Spos,1],
rxn[x1Spos,iL0N0Spos+iL0N2Sneg+iL0N4Sneg+iL0N6Spos,1],
rxn[x1Sneg,iL0N0Sneg+iL0N2Spos+iL0N4Spos+iL0N6Sneg,1],
conc[iL0N0Sneg,0.4431012961],
conc[iL0N1Sneg,0.1213298373],
conc[iL0N2Spos,0.2929079913],
conc[iL0N3Sneg,0.4845437167],
conc[iL0N4Sneg,0.6150735142],
conc[iL0N5Sneg,0.5053237287],
conc[iL0N6Sneg,0.4029008004],
conc[iL0N7Sneg,0.3415368337],
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
rxn[hL0N0Spos,iL1N1Spos+iL1N2Sneg+iL1N3Sneg+iL1N4Sneg+iL1N5Sneg,1],
rxn[hL0N0Sneg,iL1N1Sneg+iL1N2Spos+iL1N3Spos+iL1N4Spos+iL1N5Spos,1],
rxn[hL0N1Spos,iL1N0Sneg+iL1N1Spos+iL1N2Sneg+iL1N3Sneg+iL1N4Spos+iL1N5Sneg+iL1N6Spos+iL1N7Sneg,1],
rxn[hL0N1Sneg,iL1N0Spos+iL1N1Sneg+iL1N2Spos+iL1N3Spos+iL1N4Sneg+iL1N5Spos+iL1N6Sneg+iL1N7Spos,1],
rxn[hL0N2Spos,iL1N0Sneg+iL1N1Sneg+iL1N2Sneg+iL1N4Sneg+iL1N7Sneg,1],
rxn[hL0N2Sneg,iL1N0Spos+iL1N1Spos+iL1N2Spos+iL1N4Spos+iL1N7Spos,1],
rxn[hL0N3Spos,iL1N0Sneg+iL1N3Sneg+iL1N4Sneg+iL1N6Sneg+iL1N7Sneg,1],
rxn[hL0N3Sneg,iL1N0Spos+iL1N3Spos+iL1N4Spos+iL1N6Spos+iL1N7Spos,1],
rxn[hL0N4Spos,iL1N0Sneg+iL1N1Sneg+iL1N3Sneg+iL1N4Sneg+iL1N5Sneg,1],
rxn[hL0N4Sneg,iL1N0Spos+iL1N1Spos+iL1N3Spos+iL1N4Spos+iL1N5Spos,1],
rxn[hL0N5Spos,iL1N0Spos+iL1N2Sneg+iL1N6Sneg+iL1N7Sneg,1],
rxn[hL0N5Sneg,iL1N0Sneg+iL1N2Spos+iL1N6Spos+iL1N7Spos,1],
rxn[hL0N6Spos,iL1N1Spos+iL1N2Sneg+iL1N3Sneg+iL1N6Sneg+iL1N7Sneg,1],
rxn[hL0N6Sneg,iL1N1Sneg+iL1N2Spos+iL1N3Spos+iL1N6Spos+iL1N7Spos,1],
rxn[hL0N7Spos,iL1N0Sneg+iL1N2Sneg+iL1N3Sneg+iL1N4Sneg+iL1N5Sneg+iL1N6Sneg,1],
rxn[hL0N7Sneg,iL1N0Spos+iL1N2Spos+iL1N3Spos+iL1N4Spos+iL1N5Spos+iL1N6Spos,1],
conc[iL1N0Sneg,0.2597201160],
conc[iL1N1Sneg,0.6046205459],
conc[iL1N2Sneg,0.0512621680],
conc[iL1N3Sneg,0.2644855026],
conc[iL1N4Sneg,0.0188793058],
conc[iL1N5Sneg,0.7924036387],
conc[iL1N6Sneg,0.0193534885],
conc[iL1N7Sneg,0.1891287016],
rxn[iL1N0Spos,mL1N0+hL1N0Spos,1],
rxn[mL1N0+iL1N0Sneg,hL1N0Sneg,1],
rxn[iL1N1Spos,mL1N1+hL1N1Spos,1],
rxn[mL1N1+iL1N1Sneg,hL1N1Sneg,1],
rxn[iL1N2Spos,mL1N2+hL1N2Spos,1],
rxn[mL1N2+iL1N2Sneg,hL1N2Sneg,1],
rxn[iL1N3Spos,mL1N3+hL1N3Spos,1],
rxn[mL1N3+iL1N3Sneg,hL1N3Sneg,1],
rxn[iL1N4Spos,mL1N4+hL1N4Spos,1],
rxn[mL1N4+iL1N4Sneg,hL1N4Sneg,1],
rxn[iL1N5Spos,mL1N5+hL1N5Spos,1],
rxn[mL1N5+iL1N5Sneg,hL1N5Sneg,1],
rxn[iL1N6Spos,mL1N6+hL1N6Spos,1],
rxn[mL1N6+iL1N6Sneg,hL1N6Sneg,1],
rxn[iL1N7Spos,mL1N7+hL1N7Spos,1],
rxn[mL1N7+iL1N7Sneg,hL1N7Sneg,1],
rxn[hL1N0Spos,iL2N1Sneg+iL2N2Sneg+iL2N3Spos+iL2N4Spos+iL2N6Sneg+iL2N7Sneg,1],
rxn[hL1N0Sneg,iL2N1Spos+iL2N2Spos+iL2N3Sneg+iL2N4Sneg+iL2N6Spos+iL2N7Spos,1],
rxn[hL1N1Spos,iL2N0Sneg+iL2N1Sneg+iL2N3Sneg+iL2N4Sneg+iL2N5Sneg+iL2N7Sneg,1],
rxn[hL1N1Sneg,iL2N0Spos+iL2N1Spos+iL2N3Spos+iL2N4Spos+iL2N5Spos+iL2N7Spos,1],
rxn[hL1N2Spos,iL2N0Spos+iL2N1Spos+iL2N2Spos+iL2N3Spos+iL2N5Spos+iL2N6Sneg+iL2N7Sneg,1],
rxn[hL1N2Sneg,iL2N0Sneg+iL2N1Sneg+iL2N2Sneg+iL2N3Sneg+iL2N5Sneg+iL2N6Spos+iL2N7Spos,1],
rxn[hL1N3Spos,iL2N1Sneg+iL2N3Sneg+iL2N4Sneg+iL2N5Sneg+iL2N6Spos+iL2N7Spos,1],
rxn[hL1N3Sneg,iL2N1Spos+iL2N3Spos+iL2N4Spos+iL2N5Spos+iL2N6Sneg+iL2N7Sneg,1],
rxn[hL1N4Spos,iL2N0Sneg+iL2N1Sneg+iL2N2Sneg+iL2N3Spos+iL2N4Spos+iL2N5Spos+iL2N6Spos+iL2N7Spos,1],
rxn[hL1N4Sneg,iL2N0Spos+iL2N1Spos+iL2N2Spos+iL2N3Sneg+iL2N4Sneg+iL2N5Sneg+iL2N6Sneg+iL2N7Sneg,1],
rxn[hL1N5Spos,iL2N1Sneg+iL2N2Sneg+iL2N3Sneg+iL2N4Sneg+iL2N6Sneg,1],
rxn[hL1N5Sneg,iL2N1Spos+iL2N2Spos+iL2N3Spos+iL2N4Spos+iL2N6Spos,1],
rxn[hL1N6Spos,iL2N1Sneg+iL2N2Sneg+iL2N3Sneg+iL2N4Sneg+iL2N5Sneg+iL2N6Sneg+iL2N7Sneg,1],
rxn[hL1N6Sneg,iL2N1Spos+iL2N2Spos+iL2N3Spos+iL2N4Spos+iL2N5Spos+iL2N6Spos+iL2N7Spos,1],
rxn[hL1N7Spos,iL2N0Sneg+iL2N1Sneg+iL2N2Spos+iL2N3Spos+iL2N4Sneg+iL2N5Sneg+iL2N6Sneg+iL2N7Spos,1],
rxn[hL1N7Sneg,iL2N0Spos+iL2N1Spos+iL2N2Sneg+iL2N3Sneg+iL2N4Spos+iL2N5Spos+iL2N6Spos+iL2N7Sneg,1],
conc[iL2N0Sneg,0.0818248055],
conc[iL2N1Sneg,0.1497567067],
conc[iL2N2Sneg,0.9956572312],
conc[iL2N3Sneg,0.0141458276],
conc[iL2N4Sneg,0.1640850669],
conc[iL2N5Spos,1.2487780595],
conc[iL2N6Sneg,0.7726643159],
conc[iL2N7Spos,0.7908274673],
rxn[iL2N0Spos,mL2N0+hL2N0Spos,1],
rxn[mL2N0+iL2N0Sneg,hL2N0Sneg,1],
rxn[iL2N1Spos,mL2N1+hL2N1Spos,1],
rxn[mL2N1+iL2N1Sneg,hL2N1Sneg,1],
rxn[iL2N2Spos,mL2N2+hL2N2Spos,1],
rxn[mL2N2+iL2N2Sneg,hL2N2Sneg,1],
rxn[iL2N3Spos,mL2N3+hL2N3Spos,1],
rxn[mL2N3+iL2N3Sneg,hL2N3Sneg,1],
rxn[iL2N4Spos,mL2N4+hL2N4Spos,1],
rxn[mL2N4+iL2N4Sneg,hL2N4Sneg,1],
rxn[iL2N5Spos,mL2N5+hL2N5Spos,1],
rxn[mL2N5+iL2N5Sneg,hL2N5Sneg,1],
rxn[iL2N6Spos,mL2N6+hL2N6Spos,1],
rxn[mL2N6+iL2N6Sneg,hL2N6Sneg,1],
rxn[iL2N7Spos,mL2N7+hL2N7Spos,1],
rxn[mL2N7+iL2N7Sneg,hL2N7Sneg,1],
rxn[hL2N0Spos,iL3N1Sneg+iL3N2Spos+iL3N4Sneg+iL3N5Sneg+iL3N6Spos+iL3N7Spos,1],
rxn[hL2N0Sneg,iL3N1Spos+iL3N2Sneg+iL3N4Spos+iL3N5Spos+iL3N6Sneg+iL3N7Sneg,1],
rxn[hL2N1Spos,iL3N0Spos+iL3N1Spos+iL3N2Sneg+iL3N4Sneg+iL3N5Sneg+iL3N6Spos+iL3N7Spos,1],
rxn[hL2N1Sneg,iL3N0Sneg+iL3N1Sneg+iL3N2Spos+iL3N4Spos+iL3N5Spos+iL3N6Sneg+iL3N7Sneg,1],
rxn[hL2N2Spos,iL3N2Sneg+iL3N3Sneg+iL3N4Spos+iL3N5Spos+iL3N7Sneg,1],
rxn[hL2N2Sneg,iL3N2Spos+iL3N3Spos+iL3N4Sneg+iL3N5Sneg+iL3N7Spos,1],
rxn[hL2N3Spos,iL3N1Sneg+iL3N2Sneg+iL3N3Spos+iL3N4Sneg+iL3N5Spos+iL3N6Spos,1],
rxn[hL2N3Sneg,iL3N1Spos+iL3N2Spos+iL3N3Sneg+iL3N4Spos+iL3N5Sneg+iL3N6Sneg,1],
rxn[hL2N4Spos,iL3N0Spos+iL3N1Sneg+iL3N2Spos+iL3N4Sneg+iL3N5Sneg+iL3N6Sneg+iL3N7Spos,1],
rxn[hL2N4Sneg,iL3N0Sneg+iL3N1Spos+iL3N2Sneg+iL3N4Spos+iL3N5Spos+iL3N6Spos+iL3N7Sneg,1],
rxn[hL2N5Spos,iL3N0Sneg+iL3N1Sneg+iL3N3Sneg+iL3N4Sneg+iL3N6Sneg+iL3N7Sneg,1],
rxn[hL2N5Sneg,iL3N0Spos+iL3N1Spos+iL3N3Spos+iL3N4Spos+iL3N6Spos+iL3N7Spos,1],
rxn[hL2N6Spos,iL3N0Sneg+iL3N1Sneg+iL3N2Spos+iL3N3Spos+iL3N4Spos+iL3N5Spos+iL3N7Spos,1],
rxn[hL2N6Sneg,iL3N0Spos+iL3N1Spos+iL3N2Sneg+iL3N3Sneg+iL3N4Sneg+iL3N5Sneg+iL3N7Sneg,1],
rxn[hL2N7Spos,iL3N0Sneg+iL3N2Sneg+iL3N3Sneg+iL3N4Sneg+iL3N5Sneg+iL3N6Sneg+iL3N7Sneg,1],
rxn[hL2N7Sneg,iL3N0Spos+iL3N2Spos+iL3N3Spos+iL3N4Spos+iL3N5Spos+iL3N6Spos+iL3N7Spos,1],
conc[iL3N0Sneg,0.4174022576],
conc[iL3N1Sneg,0.6706942515],
conc[iL3N2Sneg,0.8279963258],
conc[iL3N3Sneg,0.0304263930],
conc[iL3N4Spos,1.6915571353],
conc[iL3N5Sneg,0.4460851056],
conc[iL3N6Sneg,0.0799063163],
conc[iL3N7Sneg,0.1689164711],
rxn[iL3N0Spos,mL3N0+hL3N0Spos,1],
rxn[mL3N0+iL3N0Sneg,hL3N0Sneg,1],
rxn[iL3N1Spos,mL3N1+hL3N1Spos,1],
rxn[mL3N1+iL3N1Sneg,hL3N1Sneg,1],
rxn[iL3N2Spos,mL3N2+hL3N2Spos,1],
rxn[mL3N2+iL3N2Sneg,hL3N2Sneg,1],
rxn[iL3N3Spos,mL3N3+hL3N3Spos,1],
rxn[mL3N3+iL3N3Sneg,hL3N3Sneg,1],
rxn[iL3N4Spos,mL3N4+hL3N4Spos,1],
rxn[mL3N4+iL3N4Sneg,hL3N4Sneg,1],
rxn[iL3N5Spos,mL3N5+hL3N5Spos,1],
rxn[mL3N5+iL3N5Sneg,hL3N5Sneg,1],
rxn[iL3N6Spos,mL3N6+hL3N6Spos,1],
rxn[mL3N6+iL3N6Sneg,hL3N6Sneg,1],
rxn[iL3N7Spos,mL3N7+hL3N7Spos,1],
rxn[mL3N7+iL3N7Sneg,hL3N7Sneg,1],
rxn[hL3N0Spos,hL4N0Spos+hL4N1Spos,1],
rxn[hL3N0Sneg,hL4N0Sneg+hL4N1Sneg,1],
rxn[hL3N1Spos,hL4N0Sneg,1],
rxn[hL3N1Sneg,hL4N0Spos,1],
rxn[hL3N2Spos,hL4N0Sneg,1],
rxn[hL3N2Sneg,hL4N0Spos,1],
rxn[hL3N3Spos,hL4N0Spos,1],
rxn[hL3N3Sneg,hL4N0Sneg,1],
rxn[hL3N4Spos,hL4N0Sneg+hL4N1Spos,1],
rxn[hL3N4Sneg,hL4N0Spos+hL4N1Sneg,1],
rxn[hL3N7Spos,hL4N0Sneg+hL4N1Sneg,1],
rxn[hL3N7Sneg,hL4N0Spos+hL4N1Spos,1],
conc[hL4N0Spos,0.8058982812],
conc[hL4N1Sneg,0.8058982812]
}

HEARTReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_] =
  reduceFFNCCrn[HEART[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], True];

cnt = Count[HEART[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[HEARTReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], rxn[___]];
Print["Number of reactions (reduced form): " <> ToString[cntReduced]];

tmax = 100;
totalCount = 0;
correctCount = 0;
matchingCount = 0;

analyzeConvergenceTime = True;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=HEART[1.529019,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p0.pdf", p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p1.pdf", p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p2.pdf", p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p3.pdf", p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p4.pdf", p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p5.pdf", p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p6.pdf", p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p7.pdf", p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p8.pdf", p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p9.pdf", p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p10.pdf", p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p11.pdf", p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p12.pdf", p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p13.pdf", p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p14.pdf", p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p15.pdf", p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p16.pdf", p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p17.pdf", p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p18.pdf", p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p19.pdf", p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p20.pdf", p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p21.pdf", p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p22.pdf", p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p23.pdf", p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p24.pdf", p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p25.pdf", p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p26.pdf", p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p27.pdf", p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p28.pdf", p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p29.pdf", p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p30.pdf", p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p31.pdf", p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p32.pdf", p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p33.pdf", p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p34.pdf", p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p35.pdf", p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p36.pdf", p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p37.pdf", p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p38.pdf", p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p39.pdf", p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p40.pdf", p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p41.pdf", p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p42.pdf", p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p43.pdf", p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p44.pdf", p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p45.pdf", p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p46.pdf", p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p47.pdf", p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p48.pdf", p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p49.pdf", p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p50.pdf", p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p51.pdf", p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p52.pdf", p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p53.pdf", p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p54.pdf", p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p55.pdf", p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p56.pdf", p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p57.pdf", p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p58.pdf", p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p59.pdf", p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p60.pdf", p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p61.pdf", p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p62.pdf", p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p63.pdf", p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p64.pdf", p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p65.pdf", p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p66.pdf", p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p67.pdf", p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p68.pdf", p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p69.pdf", p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p70.pdf", p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p71.pdf", p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p72.pdf", p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p73.pdf", p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p74.pdf", p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p75.pdf", p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p76.pdf", p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p77.pdf", p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p78.pdf", p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p79.pdf", p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p80.pdf", p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p81.pdf", p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p82.pdf", p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p83.pdf", p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p84.pdf", p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p85.pdf", p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p86.pdf", p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p87.pdf", p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p88.pdf", p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p89.pdf", p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p90.pdf", p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p91.pdf", p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p92.pdf", p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p93.pdf", p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p94.pdf", p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p95.pdf", p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p96.pdf", p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p97.pdf", p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p98.pdf", p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p99.pdf", p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["100: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p100=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p100.pdf", p100];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["101: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p101=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p101.pdf", p101];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["102: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654261939658359,-0.654261939661826];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p102.pdf", p102];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["103: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p103.pdf", p103];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["104: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p104=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p104.pdf", p104];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["105: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p105=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p105.pdf", p105];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["106: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p106=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p106.pdf", p106];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["107: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p107=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p107.pdf", p107];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["108: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p108=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p108.pdf", p108];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["109: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p109=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p109.pdf", p109];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["110: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p110=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p110.pdf", p110];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["111: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p111=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p111.pdf", p111];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["112: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p112=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p112.pdf", p112];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["113: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p113=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p113.pdf", p113];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["114: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p114=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p114.pdf", p114];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["115: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p115=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p115.pdf", p115];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["116: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p116=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p116.pdf", p116];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["117: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p117.pdf", p117];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["118: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654261939658359,-0.654261939661826];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p118.pdf", p118];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["119: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.191352121962741,-0.191352121966208];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p119=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p119.pdf", p119];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["120: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p120.pdf", p120];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["121: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p121=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p121.pdf", p121];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["122: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p122=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p122.pdf", p122];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["123: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p123=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p123.pdf", p123];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["124: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p124=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p124.pdf", p124];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["125: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p125=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p125.pdf", p125];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["126: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p126=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p126.pdf", p126];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["127: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p127=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p127.pdf", p127];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["128: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p128=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p128.pdf", p128];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["129: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p129=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p129.pdf", p129];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["130: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p130=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p130.pdf", p130];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["131: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p131=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p131.pdf", p131];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["132: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p132=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p132.pdf", p132];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["133: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p133=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p133.pdf", p133];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["134: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p134.pdf", p134];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["135: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.191352121962741,-0.191352121966208];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p135=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p135.pdf", p135];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["136: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.271557904349133, 0.271557904345666];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p136.pdf", p136];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["137: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.540733479394153,-0.54073347939762 ];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p137.pdf", p137];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["138: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p138.pdf", p138];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["139: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p139=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p139.pdf", p139];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["140: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p140=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p140.pdf", p140];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["141: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p141=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p141.pdf", p141];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["142: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p142=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p142.pdf", p142];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["143: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p143=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p143.pdf", p143];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["144: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p144=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p144.pdf", p144];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["145: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p145=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p145.pdf", p145];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["146: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p146=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p146.pdf", p146];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["147: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p147=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p147.pdf", p147];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["148: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p148=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p148.pdf", p148];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["149: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p149=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p149.pdf", p149];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["150: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p150=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p150.pdf", p150];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["151: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.540733479394153,-0.54073347939762 ];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p151=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p151.pdf", p151];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["152: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.271557904349133, 0.271557904345666];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p152=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p152.pdf", p152];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["153: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.641096905016261, 0.641096905012794];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p153=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p153.pdf", p153];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["154: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042194164323589, 0.042194164320122];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p154=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p154.pdf", p154];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["155: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.770097219419697,-0.770097219423164];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p155=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p155.pdf", p155];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["156: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p156=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p156.pdf", p156];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["157: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p157=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p157.pdf", p157];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["158: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p158=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p158.pdf", p158];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["159: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p159=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p159.pdf", p159];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["160: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p160=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p160.pdf", p160];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["161: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p161=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p161.pdf", p161];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["162: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p162=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p162.pdf", p162];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["163: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p163=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p163.pdf", p163];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["164: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p164=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p164.pdf", p164];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["165: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p165=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p165.pdf", p165];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["166: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p166=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p166.pdf", p166];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["167: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.770097219419697,-0.770097219423164];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p167=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p167.pdf", p167];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["168: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042194164323589, 0.042194164320122];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p168=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p168.pdf", p168];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["169: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.641096905016261, 0.641096905012794];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p169=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p169.pdf", p169];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["170: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p170=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p170.pdf", p170];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["171: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.697861269258815, 0.697861269255348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p171=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p171.pdf", p171];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["172: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155722892808697, 0.155722892805229];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p172=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p172.pdf", p172];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["173: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.656568550539234,-0.656568550542702];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p173=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p173.pdf", p173];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["174: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p174=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p174.pdf", p174];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["175: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p175=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p175.pdf", p175];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["176: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p176=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p176.pdf", p176];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["177: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p177=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p177.pdf", p177];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["178: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p178=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p178.pdf", p178];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["179: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p179=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p179.pdf", p179];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["180: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p180=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p180.pdf", p180];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["181: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p181=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p181.pdf", p181];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["182: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p182=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p182.pdf", p182];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["183: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.656568550539234,-0.656568550542702];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p183=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p183.pdf", p183];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["184: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155722892808697, 0.155722892805229];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p184=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p184.pdf", p184];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["185: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.697861269258815, 0.697861269255348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p185=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p185.pdf", p185];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["186: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p186=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p186.pdf", p186];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["187: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p187=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p187.pdf", p187];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["188: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p188=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p188.pdf", p188];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["189: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.754625156664211, 0.754625156660743];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p189=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p189.pdf", p189];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["190: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.269250608014843, 0.269250608011376];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p190=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p190.pdf", p190];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["191: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.543040805530766,-0.543040805534233];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p191=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p191.pdf", p191];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["192: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p192=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p192.pdf", p192];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["193: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p193=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p193.pdf", p193];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["194: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p194=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p194.pdf", p194];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["195: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p195=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p195.pdf", p195];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["196: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p196=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p196.pdf", p196];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["197: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p197=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p197.pdf", p197];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["198: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p198=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p198.pdf", p198];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["199: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.543040805530766,-0.543040805534233];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p199=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p199.pdf", p199];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["200: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.269250608014843, 0.269250608011376];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p200=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p200.pdf", p200];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["201: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.754625156664211, 0.754625156660743];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p201=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p201.pdf", p201];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["202: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p202=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p202.pdf", p202];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["203: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p203=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p203.pdf", p203];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["204: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p204=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p204.pdf", p204];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["205: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132814, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p205=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p205.pdf", p205];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["206: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p206=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p206.pdf", p206];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["207: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.811389371895152, 0.811389371891685];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p207=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p207.pdf", p207];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["208: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.382779068279049, 0.382779068275582];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p208=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p208.pdf", p208];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["209: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.42951234526656 ,-0.429512345270027];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p209=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p209.pdf", p209];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["210: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p210=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p210.pdf", p210];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["211: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p211=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p211.pdf", p211];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["212: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p212=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p212.pdf", p212];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["213: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p213=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p213.pdf", p213];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["214: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p214=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p214.pdf", p214];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["215: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.42951234526656 ,-0.429512345270027];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p215=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p215.pdf", p215];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["216: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.382779068279049, 0.382779068275582];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p216=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p216.pdf", p216];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["217: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.811389371895152, 0.811389371891685];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p217=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p217.pdf", p217];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["218: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p218=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p218.pdf", p218];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["219: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132814, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p219=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p219.pdf", p219];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["220: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p220=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p220.pdf", p220];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["221: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p221=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p221.pdf", p221];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["222: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p222=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p222.pdf", p222];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["223: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p223=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p223.pdf", p223];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["224: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p224=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p224.pdf", p224];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["225: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.868153363608676, 0.868153363605209];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p225=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p225.pdf", p225];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["226: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.462007656835872, 0.462007656832405];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p226=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p226.pdf", p226];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["227: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.31598433203719 ,-0.315984332040657];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p227=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p227.pdf", p227];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["228: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p228=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p228.pdf", p228];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["229: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p229=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p229.pdf", p229];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["230: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p230=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p230.pdf", p230];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["231: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.31598433203719 ,-0.315984332040657];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p231=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p231.pdf", p231];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["232: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.462007656835872, 0.462007656832405];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p232=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p232.pdf", p232];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["233: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.868153363608676, 0.868153363605209];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p233=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p233.pdf", p233];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["234: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p234=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p234.pdf", p234];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["235: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p235=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p235.pdf", p235];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["236: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p236=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p236.pdf", p236];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["237: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p237=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p237.pdf", p237];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["238: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p238=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p238.pdf", p238];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["239: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p239=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p239.pdf", p239];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["240: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p240=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p240.pdf", p240];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["241: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p241=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p241.pdf", p241];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["242: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p242=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p242.pdf", p242];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["243: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p243=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p243.pdf", p243];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["244: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.518771901869136, 0.518771901865669];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p244=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p244.pdf", p244];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["245: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.071379477798024, 0.071379477794557];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p245=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p245.pdf", p245];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.720147,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["246: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.714360735281804,-0.714360735285272];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p246=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p246.pdf", p246];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,1.314001,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["247: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.071379477798024, 0.071379477794557];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p247=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p247.pdf", p247];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.907855,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["248: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.518771901869136, 0.518771901865669];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p248=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p248.pdf", p248];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.501709,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["249: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p249=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p249.pdf", p249];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0,0.095564,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["250: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p250=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p250.pdf", p250];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.310582,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["251: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p251=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p251.pdf", p251];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[0.716728,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["252: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p252=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p252.pdf", p252];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.122873,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["253: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p253=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p253.pdf", p253];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

rsys=HEART[1.529019,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["254: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p254=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEART_p254.pdf", p254];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEART_convergence_times.csv", ","], WriteLine["HEART_convergence_times.csv", ""]];
  ];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
tmax = 100;
totalCount = 0;
correctCount = 0;
matchingCount = 0;

analyzeConvergenceTime = False;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=HEARTReduced[1.529019,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p0.pdf", p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p1.pdf", p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p2.pdf", p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p3.pdf", p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p4.pdf", p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p5.pdf", p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p6.pdf", p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p7.pdf", p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p8.pdf", p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p9.pdf", p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p10.pdf", p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p11.pdf", p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p12.pdf", p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p13.pdf", p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p14.pdf", p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p15.pdf", p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p16.pdf", p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p17.pdf", p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p18.pdf", p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p19.pdf", p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p20.pdf", p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p21.pdf", p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p22.pdf", p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p23.pdf", p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p24.pdf", p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p25.pdf", p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p26.pdf", p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p27.pdf", p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p28.pdf", p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p29.pdf", p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p30.pdf", p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p31.pdf", p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p32.pdf", p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,1.388730];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p33.pdf", p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p34.pdf", p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p35.pdf", p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p36.pdf", p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p37.pdf", p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p38.pdf", p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p39.pdf", p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p40.pdf", p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p41.pdf", p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p42.pdf", p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p43.pdf", p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p44.pdf", p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p45.pdf", p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p46.pdf", p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p47.pdf", p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p48.pdf", p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p49.pdf", p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,1.157275];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p50.pdf", p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p51.pdf", p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p52.pdf", p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p53.pdf", p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p54.pdf", p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p55.pdf", p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p56.pdf", p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p57.pdf", p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p58.pdf", p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p59.pdf", p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p60.pdf", p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p61.pdf", p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p62.pdf", p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p63.pdf", p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p64.pdf", p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p65.pdf", p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p66.pdf", p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.925820];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p67.pdf", p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p68.pdf", p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p69.pdf", p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p70.pdf", p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p71.pdf", p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p72.pdf", p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p73.pdf", p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p74.pdf", p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p75.pdf", p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p76.pdf", p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p77.pdf", p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p78.pdf", p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p79.pdf", p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p80.pdf", p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p81.pdf", p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p82.pdf", p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p83.pdf", p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.694365];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p84.pdf", p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p85.pdf", p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p86.pdf", p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p87.pdf", p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p88.pdf", p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p89.pdf", p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p90.pdf", p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p91.pdf", p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p92.pdf", p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p93.pdf", p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p94.pdf", p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p95.pdf", p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p96.pdf", p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p97.pdf", p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p98.pdf", p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p99.pdf", p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["100: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p100=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p100.pdf", p100];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.462910];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["101: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p101=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p101.pdf", p101];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["102: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654261939658359,-0.654261939661826];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p102.pdf", p102];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["103: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p103.pdf", p103];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["104: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p104=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p104.pdf", p104];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["105: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p105=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p105.pdf", p105];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["106: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p106=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p106.pdf", p106];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["107: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p107=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p107.pdf", p107];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["108: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p108=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p108.pdf", p108];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["109: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p109=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p109.pdf", p109];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["110: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p110=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p110.pdf", p110];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["111: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p111=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p111.pdf", p111];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["112: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p112=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p112.pdf", p112];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["113: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p113=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p113.pdf", p113];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["114: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p114=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p114.pdf", p114];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["115: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p115=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p115.pdf", p115];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["116: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p116=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p116.pdf", p116];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["117: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p117.pdf", p117];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,0.231455];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["118: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654261939658359,-0.654261939661826];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p118.pdf", p118];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["119: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.191352121962741,-0.191352121966208];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p119=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p119.pdf", p119];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["120: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p120.pdf", p120];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["121: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p121=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p121.pdf", p121];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["122: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p122=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p122.pdf", p122];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["123: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p123=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p123.pdf", p123];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["124: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p124=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p124.pdf", p124];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["125: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p125=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p125.pdf", p125];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["126: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p126=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p126.pdf", p126];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["127: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p127=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p127.pdf", p127];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["128: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p128=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p128.pdf", p128];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["129: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p129=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p129.pdf", p129];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["130: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p130=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p130.pdf", p130];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["131: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p131=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p131.pdf", p131];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["132: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p132=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p132.pdf", p132];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["133: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p133=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p133.pdf", p133];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["134: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p134.pdf", p134];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0,-0.000000];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["135: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.191352121962741,-0.191352121966208];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p135=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p135.pdf", p135];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["136: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.271557904349133, 0.271557904345666];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p136.pdf", p136];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["137: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.540733479394153,-0.54073347939762 ];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p137.pdf", p137];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["138: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p138.pdf", p138];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["139: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p139=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p139.pdf", p139];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["140: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p140=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p140.pdf", p140];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["141: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p141=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p141.pdf", p141];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["142: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p142=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p142.pdf", p142];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["143: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p143=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p143.pdf", p143];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["144: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p144=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p144.pdf", p144];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["145: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p145=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p145.pdf", p145];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["146: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p146=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p146.pdf", p146];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["147: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p147=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p147.pdf", p147];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["148: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p148=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p148.pdf", p148];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["149: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p149=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p149.pdf", p149];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["150: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p150=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p150.pdf", p150];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["151: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.540733479394153,-0.54073347939762 ];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p151=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p151.pdf", p151];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.231455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["152: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.271557904349133, 0.271557904345666];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p152=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p152.pdf", p152];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["153: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.641096905016261, 0.641096905012794];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p153=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p153.pdf", p153];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["154: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042194164323589, 0.042194164320122];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p154=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p154.pdf", p154];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["155: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.770097219419697,-0.770097219423164];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p155=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p155.pdf", p155];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["156: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p156=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p156.pdf", p156];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["157: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p157=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p157.pdf", p157];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["158: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p158=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p158.pdf", p158];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["159: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p159=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p159.pdf", p159];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["160: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p160=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p160.pdf", p160];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["161: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p161=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p161.pdf", p161];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["162: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p162=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p162.pdf", p162];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["163: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p163=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p163.pdf", p163];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["164: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p164=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p164.pdf", p164];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["165: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p165=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p165.pdf", p165];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["166: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p166=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p166.pdf", p166];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["167: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.770097219419697,-0.770097219423164];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p167=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p167.pdf", p167];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["168: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042194164323589, 0.042194164320122];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p168=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p168.pdf", p168];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.462910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["169: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.641096905016261, 0.641096905012794];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p169=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p169.pdf", p169];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["170: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p170=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p170.pdf", p170];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["171: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.697861269258815, 0.697861269255348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p171=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p171.pdf", p171];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["172: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155722892808697, 0.155722892805229];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p172=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p172.pdf", p172];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["173: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.656568550539234,-0.656568550542702];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p173=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p173.pdf", p173];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["174: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p174=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p174.pdf", p174];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["175: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p175=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p175.pdf", p175];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["176: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p176=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p176.pdf", p176];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["177: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p177=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p177.pdf", p177];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["178: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p178=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p178.pdf", p178];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["179: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p179=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p179.pdf", p179];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["180: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p180=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p180.pdf", p180];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["181: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p181=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p181.pdf", p181];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["182: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p182=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p182.pdf", p182];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["183: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.656568550539234,-0.656568550542702];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p183=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p183.pdf", p183];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["184: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155722892808697, 0.155722892805229];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p184=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p184.pdf", p184];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["185: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.697861269258815, 0.697861269255348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p185=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p185.pdf", p185];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.694365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["186: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p186=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p186.pdf", p186];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["187: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p187=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p187.pdf", p187];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["188: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p188=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p188.pdf", p188];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["189: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.754625156664211, 0.754625156660743];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p189=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p189.pdf", p189];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["190: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.269250608014843, 0.269250608011376];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p190=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p190.pdf", p190];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["191: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.543040805530766,-0.543040805534233];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p191=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p191.pdf", p191];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["192: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p192=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p192.pdf", p192];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["193: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p193=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p193.pdf", p193];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["194: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p194=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p194.pdf", p194];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["195: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p195=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p195.pdf", p195];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["196: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p196=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p196.pdf", p196];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["197: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214822];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p197=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p197.pdf", p197];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["198: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p198=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p198.pdf", p198];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["199: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.543040805530766,-0.543040805534233];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p199=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p199.pdf", p199];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["200: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.269250608014843, 0.269250608011376];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p200=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p200.pdf", p200];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["201: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.754625156664211, 0.754625156660743];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p201=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p201.pdf", p201];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["202: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p202=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p202.pdf", p202];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,0.925820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["203: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p203=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p203.pdf", p203];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["204: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p204=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p204.pdf", p204];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["205: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132814, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p205=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p205.pdf", p205];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["206: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p206=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p206.pdf", p206];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["207: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.811389371895152, 0.811389371891685];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p207=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p207.pdf", p207];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["208: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.382779068279049, 0.382779068275582];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p208=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p208.pdf", p208];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["209: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.42951234526656 ,-0.429512345270027];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p209=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p209.pdf", p209];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["210: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p210=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p210.pdf", p210];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["211: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p211=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p211.pdf", p211];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["212: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p212=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p212.pdf", p212];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["213: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p213=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p213.pdf", p213];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["214: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p214=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p214.pdf", p214];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["215: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.42951234526656 ,-0.429512345270027];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p215=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p215.pdf", p215];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["216: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.382779068279049, 0.382779068275582];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p216=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p216.pdf", p216];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["217: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.811389371895152, 0.811389371891685];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p217=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p217.pdf", p217];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["218: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p218=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p218.pdf", p218];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["219: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132814, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p219=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p219.pdf", p219];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,1.157275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["220: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p220=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p220.pdf", p220];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["221: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p221=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p221.pdf", p221];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["222: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p222=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p222.pdf", p222];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["223: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p223=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p223.pdf", p223];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["224: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p224=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p224.pdf", p224];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["225: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.868153363608676, 0.868153363605209];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p225=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p225.pdf", p225];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["226: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.462007656835872, 0.462007656832405];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p226=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p226.pdf", p226];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["227: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.31598433203719 ,-0.315984332040657];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p227=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p227.pdf", p227];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["228: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p228=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p228.pdf", p228];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["229: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p229=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p229.pdf", p229];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["230: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.805898281211354,-0.805898281214821];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p230=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p230.pdf", p230];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["231: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.31598433203719 ,-0.315984332040657];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p231=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p231.pdf", p231];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["232: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.462007656835872, 0.462007656832405];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p232=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p232.pdf", p232];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["233: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.868153363608676, 0.868153363605209];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p233=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p233.pdf", p233];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["234: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p234=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p234.pdf", p234];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["235: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p235=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p235.pdf", p235];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["236: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p236=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p236.pdf", p236];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,1.388729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["237: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p237=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p237.pdf", p237];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["238: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p238=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p238.pdf", p238];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["239: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p239=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p239.pdf", p239];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["240: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p240=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p240.pdf", p240];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["241: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p241=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p241.pdf", p241];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["242: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p242=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p242.pdf", p242];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["243: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p243=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p243.pdf", p243];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["244: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.518771901869136, 0.518771901865669];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p244=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p244.pdf", p244];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["245: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.071379477798024, 0.071379477794557];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p245=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p245.pdf", p245];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.720147,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["246: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.714360735281804,-0.714360735285272];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p246=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p246.pdf", p246];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,1.314001,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["247: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.071379477798024, 0.071379477794557];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p247=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p247.pdf", p247];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.907855,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["248: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.518771901869136, 0.518771901865669];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p248=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p248.pdf", p248];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.501709,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["249: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p249=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p249.pdf", p249];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0,0.095564,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["250: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129348];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p250=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p250.pdf", p250];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.310582,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["251: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p251=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p251.pdf", p251];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[0.716728,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["252: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p252=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p252.pdf", p252];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.122873,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["253: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p253=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p253.pdf", p253];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

rsys=HEARTReduced[1.529019,0,1.620184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL4N0Spos[tmax] - hL4N0Sneg[tmax])/.sol,(hL4N1Spos[tmax] - hL4N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["254: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.885658854132815, 0.885658854129347];
plotter={(hL4N0Spos[t] - hL4N0Sneg[t])/.sol,(hL4N1Spos[t] - hL4N1Sneg[t])/.sol};
p254=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export["../results/HEARTReduced_p254.pdf", p254];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    If[i < Length[times], WriteString["HEARTReduced_convergence_times.csv", ","], WriteLine["HEARTReduced_convergence_times.csv", ""]];
  ];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
