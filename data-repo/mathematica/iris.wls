#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

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
rxn[x1Spos,iL0N2Sneg,1],
rxn[x1Sneg,iL0N2Spos,1],
rxn[x2Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x2Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
rxn[x3Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x3Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
conc[iL0N0Sneg,0.7749611705],
conc[iL0N1Sneg,0.9962031582],
conc[iL0N2Spos,0.5672149702],
rxn[iL0N0Spos,mL0N0+hL0N0Spos,1],
rxn[mL0N0+iL0N0Sneg,hL0N0Sneg,1],
rxn[iL0N1Spos,mL0N1+hL0N1Spos,1],
rxn[mL0N1+iL0N1Sneg,hL0N1Sneg,1],
rxn[iL0N2Spos,mL0N2+hL0N2Spos,1],
rxn[mL0N2+iL0N2Sneg,hL0N2Sneg,1],
rxn[hL0N0Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N0Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N1Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N1Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N2Spos,hL1N0Sneg+hL1N1Spos+hL1N2Spos,1],
rxn[hL0N2Sneg,hL1N0Spos+hL1N1Sneg+hL1N2Sneg,1],
conc[hL1N0Spos,0.5796289994],
conc[hL1N1Sneg,0.4018247924],
conc[hL1N2Sneg,1.1487628737]
}

IRISReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_] =
  reduceFFNCCrn[IRIS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], keepInputLayer=True, inputPrefix="x"];

cnt = Count[IRIS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[IRISReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], rxn[___]];
Print["Number of reactions (reduced form): " <> ToString[cntReduced]];

tmax = 50;
totalCount = 0;
correctCount = 0;
matchingCount = 0;

analyzeConvergenceTime = False;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=IRIS[0,0.900681,1.032057,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.586988,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.143017,0,0.124958,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.922224,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.385353,0.337848,0,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.434316,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.506521,0.106445,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.210349,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,1.263460,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.939558,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.537178,1.957669,0,0,1.170675,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.715553,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.506521,0.800654,0,0,1.341272,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.829951,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0.800654,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.419887,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.748856,0,0.356361,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.759466,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.143017,0.106445,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.978319,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.537178,1.494863,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.629424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.264185,0.800654,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.605357,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.264185,0,0.124958,0,1.341272,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.174864,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.870024,0,0.124958,0,1.511870,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.951301,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,2.189072,0,0,1.455004,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.009559,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,3.114684,0,0,1.284407,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.622795,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.537178,1.957669,0,0,1.398138,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.943016,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,1.032057,0,0,1.341272,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.455515,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,1.726266,0,0,1.170675,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.252119,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,1.726266,0,0,1.284407,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.092858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.537178,0.800654,0,0,1.170675,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.821484,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,1.494863,0,0,1.284407,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.729982,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.506521,1.263460,0,0,1.568735,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.651693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,0.569251,0,0,1.170675,0,0.918558];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.559166,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.264185,0.800654,0,0,1.056944,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.434760,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0,0.124958,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.687324,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0.800654,0,0,1.227541,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.100075,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.779513,1.032057,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.408954,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.779513,0.800654,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.234417,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.385353,0.337848,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.263719,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.264185,0.106445,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.911148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.537178,0.800654,0,0,1.284407,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.672269,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.779513,2.420475,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.928845,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,2.651878,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.722137,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.143017,0.106445,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.978319,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0.337848,0,0,1.455004,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.127678,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,1.032057,0,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.159182,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.143017,0.106445,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.978319,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.748856,0,0.124958,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.584929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,0.800654,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.298719,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,1.032057,0,0,1.398138,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.633548,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.627688,0,1.744778,0,1.398138,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.952109,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.748856,0.337848,0,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.797819,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,1.032057,0,0,1.227541,0,0.787085];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.068532,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,1.726266,0,0,1.056944,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.733922,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.264185,0,0.124958,0,1.341272,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.911919,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,1.726266,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.167465,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.506521,0.337848,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.498618,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.658345,1.494863,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.750592,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0.569251,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.245349,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.401508,0,0.337848,0,0.535296,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.474765951003068, 0.602503019232043,-0.094367923327578];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.539351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0.337848,0,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.510249075084806, 0.602503019232043,-0.05888479924584 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.830086,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.280340,0,0.106445,0,0.649027,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.245573631759551, 0.784910024735424, 0.676439757428905];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.431985,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,0,1.744778,0.137236,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.002826169873429, 2.180630376850361, 1.433692295542782];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.431250,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.795669,0,0,0.587764,0.478430,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.549592343591524, 1.528114879760256, 0.980458469260878];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.258035,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,0,0.587764,0.421564,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.130139864159424, 1.307944071136357, 0.561005989828778];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.316228,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0.569251,0,0.535296,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.835992387618052, 0.30436253243045 , 0.266858513287406];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.185525,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.143017,0,1.513375,0,0.260824,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.978944366758487, 1.15674857373542 , 0.409810492427841];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.178409,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.916837,0,0,0.356361,0.478430,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.955602603016623, 1.133406809993556, 0.386468728685977];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.884854,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.779513,0,0.819166,0.080370,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.151821575375336, 1.329625782352269, 0.582687701044689];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.943749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0,2.438987,0,0.147093,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.018287635265205, 2.196091842242137, 1.449153760934558];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.869122,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.068662,0,0,0.124958,0.250967,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.759682761966308, 0.937486968943241, 0.190548887635661];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "0.840759,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.189830,0,0,1.976181,0.137236,0,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.83981022494736 , 2.017614431924293, 1.270676350616714];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.564440,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.310998,0,0,0.356361,0.535296,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.168974846399228, 1.296711914628203, 0.599840972068582];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.467353,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.294842,0,0.356361,0,0.090227,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.3869455631241  , 0.564749770101033,-0.182188311206547];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "0.874655,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0.106445,0,0.364699,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.510538234793385, 0.688342441770318,-0.058595639537261];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.773848,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.294842,0,0.124958,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.973055005348913, 1.06530894949615 , 0.403921131018266];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.237536,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,0,0.819166,0.194102,0,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.739661103276346, 0.917465310253279, 0.1705272289457  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.326967,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.432165,0,0,1.976181,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.824278726405337, 2.916532670552574, 2.255144852074691];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.226083,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.294842,0,1.281972,0.080370,0,0,0.129720];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.220208595581322, 1.398012802558255, 0.651074721250676];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.786904,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.068662,0,0.337848,0,0.592162,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 1 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.026830284834102, 0.215953853534502, 1.457696410503456];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.789262,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.310998,0,0,0.587764,0.137236,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.845811344213163, 1.023615551190096, 0.276677469882516];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.169223,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0,1.281972,0.649027,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.633991422551871, 2.173327815527743, 2.064857548221224];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.880505,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.310998,0,0,0.587764,0.535296,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.112398302118555, 1.290202509095488, 0.543264427787909];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.435810,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0,0.356361,0.307833,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.785005491048866, 0.962809698025799, 0.21587161671822 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.471921,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.916837,0,0,0.124958,0.364699,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.741941199925439, 0.919745406902372, 0.172807325594792];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.671192,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.159173,0,0,0.587764,0.592162,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.514109219509786, 1.528114879760256, 0.94497534517914 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.603797,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0,0.124958,0.705893,0,0.659118,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 1 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.436411528975603, 0.696501345839478, 1.867277654644956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.527973,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.189830,0,0,0.356361,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.204457970480966, 1.296711914628203, 0.63532409615032 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.363927,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,0,1.050569,0,0.147093,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.629869844472885, 0.807674051449818, 0.060735970142239];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.632529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,0,1.513375,0.023504,0,0,0.129720];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.394745856724123, 1.572550063701056, 0.825611982393476];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.082610,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,0,1.513375,0,0.033361,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.206407182715497, 1.38421138969243 , 0.63727330838485 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.223939,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,0,0.819166,0.080370,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.888875635336589, 1.066679842313522, 0.319741761005942];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "0.953796,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.189830,0,0,0.819166,0.762759,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 1 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.906798626281399, 1.465317507265758, 2.337664751950752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.299400,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.537178,0,0.124958,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.973055005348913, 1.06530894949615 , 0.403921131018266];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.479871,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.189830,0,0.800654,0,0.421564,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.310389084859448, 0.139697088967937,-0.258744789471198];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.939693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0.106445,0,0.535296,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.969114856173869, 0.833905984364096, 0.399980981843223];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.075918,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0,1.744778,0.364699,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.230288985830438, 2.408093192807371, 1.661155111499792];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.796036,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.294842,0,0.124958,0.194102,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.439871117938308, 0.617675324915241,-0.129262756392339];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "0.747127,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,0,1.281972,0.137236,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.540020239609322, 1.717824446586255, 0.970886365278675];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.968444,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.416010,0,1.050569,0.364699,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.404607120414904, 1.582411327391837, 0.835473246084257];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.833031,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.310998,0,0,0.124958,0.478430,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.855672607903943, 1.033476814880876, 0.286538733573297];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.179084,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,0,1.050569,0.137236,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.177144304457894, 1.354948511434827, 0.608010430127248];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.242064,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.021849,0,1.744778,0,0.260824,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.210347331890541, 1.388151538867474, 0.641213457559894];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.288644,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.294842,0,0.819166,0.250967,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.190945717323719, 1.368749924300652, 0.621811842993073];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.498202,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,0,0.124958,0.250967,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.365263851908187, 0.54306805888512 ,-0.20387002242246 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "0.551352,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,0,0.356361,0.250967,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.728139787059614, 0.905943994036547, 0.159005912728968];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "0.914228,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.432165,0,0,0.356361,0.307833,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.785005491048866, 0.962809698025799, 0.21587161671822 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.229585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.900681,0,1.281972,0,0.431421,0,0.129720];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.708417259678051, 0.886221466654984, 0.139283385347404];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.743795,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,0,0.587764,0.194102,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.902677048202415, 1.080481255179348, 0.333543173871768];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.088765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0.569251,0,1.274550,0,1.710902,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["100: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.603525033721982,-1.618148349604194, 6.034391159391336];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p100=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p100.svg"}], p100];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.108036,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,0,0.819166,0.762759,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["101: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.09005535645576 , 1.070898597207637, 3.520921482125114];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p101=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p101.svg"}], p101];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.556495,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.522676,0,0,0.124958,1.217684,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["102: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.549461176917902,-0.341181870141289, 4.980327302587256];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p102.svg"}], p102];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.050328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0,0.356361,1.047087,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["103: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.085816075972319, 0.455237117016644, 3.516682201641673];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p103.svg"}], p103];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.747372,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.795669,0,0,0.124958,1.160819,0,1.316483,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["104: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.773282975008264,-0.41578913617141 , 5.204149100677618];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p104=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p104.svg"}], p104];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.397928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[2.128516,0,0,0.124958,1.615744,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["105: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.7436409606922  ,-0.739241798066054, 6.174507086361554];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p105=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p105.svg"}], p105];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.054227,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,1.143017,0,1.281972,0.421564,0,0.659118,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["106: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.740440794797084, 2.137844691446006, 2.171306920466437];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p106=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p106.svg"}], p106];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.505672,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.765012,0,0,0.356361,1.445147,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["107: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.279995859746619, 0.057177189091877, 4.710861985415973];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p107=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p107.svg"}], p107];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.357110,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0,1.281972,1.160819,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["108: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.352622160436047, 1.267117569566351, 4.783488286105401];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p108=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p108.svg"}], p108];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.271387,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.643844,0,1.263460,0,1.331416,0,1.710902,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["109: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.07991325029358 ,-2.369222948989608, 5.510779375962934];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p109=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p109.svg"}], p109];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.949622,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.795669,0,0.337848,0,0.762759,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["110: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.327459440853615,-0.217589198472002, 2.758325566522969];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p110=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p110.svg"}], p110];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.949813,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0,0.819166,0.876490,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["111: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.431249580391274, 0.957167189229133, 3.862115706060628];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p111=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p111.svg"}], p111];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.292221,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.159173,0,0,0.124958,0.990221,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["112: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.867072729046872,-0.113719054184278, 4.297938854716226];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p112=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p112.svg"}], p112];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.459361,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.173674,0,1.281972,0.705893,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["113: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.776683084810231, 1.459097261441623, 4.207549210479585];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p113=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p113.svg"}], p113];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.215076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,0,0.587764,0.762759,0,1.579429,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["114: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.830746941614312, 0.182130781978717, 5.261613067283666];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p114=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p114.svg"}], p114];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.982457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0.337848,0,0.876490,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["115: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.851910394963492,-0.725739516508627, 4.282776520632846];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p115=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p115.svg"}], p115];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.336795,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.795669,0,0,0.124958,0.990221,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["116: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.683815998872509, 0.280699855873842, 3.114682124541863];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p116=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p116.svg"}], p116];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.701439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[2.249683,0,1.726266,0,1.672610,0,1.316483,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["117: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.457433261661653,-2.778804193131106, 4.888299387331007];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p117.svg"}], p117];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "6.965042,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[2.249683,0,0,1.050569,1.786341,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["118: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-8.969881977239927,-0.247172989544347, 8.40074810290928 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p118.svg"}], p118];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "6.534550,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.189830,0,0,1.976181,0.705893,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["119: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.498797429915785, 2.810671006934649, 2.929663555585139];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p119=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p119.svg"}], p119];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.268076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.280340,0,0.337848,0,1.103953,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["120: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.534298842834521,-0.953202332465637, 4.965164968503875];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p120.svg"}], p120];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.170097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.294842,0,0.587764,0.649027,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["121: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.911877077446316, 0.821754070034716, 3.34274320311567 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p121=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p121.svg"}], p121];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.585169,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[2.249683,0,0,0.587764,1.672610,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["122: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.982625092865944,-0.201828601771826, 6.413491218535298];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p122=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p122.svg"}], p122];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.563594,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0,0.819166,0.649027,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["123: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.354442222462127, 1.316102975205515, 2.785308348131481];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p123=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p123.svg"}], p123];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.812118,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0.569251,0,1.103953,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["124: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.514058057586229,-0.921659357558943, 3.944924183255583];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p124=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p124.svg"}], p124];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.896219,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.643844,0,0.337848,0,1.274550,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["125: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.073995628447189,-0.466434594336526, 3.504861754116543];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p125=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p125.svg"}], p125];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.046833,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.432165,0,0,0.587764,0.592162,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["126: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.952442145362316, 1.141565714062716, 2.38330827103167 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p126=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p126.svg"}], p126];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.402681,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.310998,0,0,0.124958,0.649027,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["127: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.660233327065968, 0.621894079809356, 2.091099452735321];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p127=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p127.svg"}], p127];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.875573,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0,0.587764,1.047087,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["128: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.500475771278736, 0.292221172090576, 4.93134189694809 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p128=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p128.svg"}], p128];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.494362,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.643844,0,0,0.124958,1.160819,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["129: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.40676951465954 , 0.373048683944832, 2.837635640328893];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p129=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p129.svg"}], p129];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.457265,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.886180,0,0,0.587764,1.331416,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["130: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.564623511001279, 0.270838592183061, 4.995489636670633];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p130=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p130.svg"}], p130];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.727423,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[2.492019,0,1.726266,0,1.502013,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["131: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.156804105642141,-2.345261141124602, 3.587670231311495];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p131=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p131.svg"}], p131];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "6.773835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0,0.587764,1.047087,0,1.316483,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["132: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.894894681336857, 0.160748202071202, 5.325760807006211];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p132=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p132.svg"}], p132];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.625835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0,0.587764,0.762759,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["133: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.280976751091225, 1.365387512153079, 1.711842876760578];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p133=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p133.svg"}], p133];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.300027,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.310998,0,0,1.050569,1.047087,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["134: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.202349331135994, 1.675337892490297, 2.633215456805348];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p134.svg"}], p134];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.673353,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[2.249683,0,0,0.124958,1.331416,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["135: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.679493220969657,-0.717859218158541, 6.110359346639011];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p135=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p135.svg"}], p135];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "5.154012,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0.800654,0,1.047087,0,1.579429,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["136: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.295314710660779,-1.490615528759864, 4.726180836330133];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p136.svg"}], p136];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.980503,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.674501,0,0.106445,0,0.990221,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["137: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.452413033740457, 0.049296890741789, 2.883279159409811];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p137.svg"}], p137];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.561759,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.189830,0,0,0.124958,0.592162,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["138: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.489636215098209, 0.678759783798609, 1.920502340767563];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p138.svg"}], p138];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.697540,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.280340,0,0.106445,0,0.933356,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["139: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.465072651947063,-0.28825631532708 , 3.895938777616417];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p139=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p139.svg"}], p139];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.505151,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0.106445,0,1.047087,0,1.579429,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["140: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.989523606056937,-0.796406633363705, 5.420389731726291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p140=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p140.svg"}], p140];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.770966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.280340,0,0.106445,0,0.762759,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["141: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.742119136160031,-0.38060514339807 , 4.172985261829385];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p141=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p141.svg"}], p141];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.597500,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0,0.052506,0,0.819166,0.762759,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["142: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.09005535645576 , 1.070898597207637, 3.520921482125114];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p142=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p142.svg"}], p142];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.556495,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.159173,0,0.337848,0,1.217684,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["143: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.875493066770036,-1.066933740444142, 5.30635919243939 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p143=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p143.svg"}], p143];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.162661,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0.569251,0,1.103953,0,1.710902,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["144: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.091733697818712,-1.447551237636437, 5.522599823488066];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p144=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p144.svg"}], p144];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "4.422111,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[1.038005,0,0,0.124958,0.819624,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["145: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.144119213259843,-0.206067882255269, 4.574985338929197];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p145=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p145.svg"}], p145];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.430542,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.553333,0,0,1.281972,0.705893,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["146: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.382264174752111, 1.590570231460996, 3.813130300421465];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p146=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p146.svg"}], p146];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.463262,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.795669,0,0,0.124958,0.819624,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["147: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.960862483085481, 0.188351027802853, 3.391728608754835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p147=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p147.svg"}], p147];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "2.793788,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.432165,0,0.800654,0,0.933356,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["148: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.559701576667145,-1.245411150761986, 3.990567702336499];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p148=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p148.svg"}], p148];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "3.614131,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

rsys=IRIS[0.068662,0,0,0.124958,0.762759,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["149: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.001427551001481, 0.508162671830852, 2.432293676670835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p149=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRIS_p149.svg"}], p149];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRIS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRIS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRIS_convergence_times.csv", ","];
  ];
  WriteString["IRIS_convergence_times.csv", "1.746969,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRIS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRIS_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
tmax = 50;
totalCount = 0;
correctCount = 0;
matchingCount = 0;

analyzeConvergenceTime = False;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=IRISReduced[0,0.900681,1.032057,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.586988,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.143017,0,0.124958,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.922224,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.385353,0.337848,0,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.434316,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.506521,0.106445,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.210349,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,1.263460,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.939558,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.537178,1.957669,0,0,1.170675,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.715553,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.506521,0.800654,0,0,1.341272,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.829951,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0.800654,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.419887,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.748856,0,0.356361,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.759466,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.143017,0.106445,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.978319,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.537178,1.494863,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.629424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.264185,0.800654,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.605357,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.264185,0,0.124958,0,1.341272,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.174864,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.870024,0,0.124958,0,1.511870,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.951301,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,2.189072,0,0,1.455004,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.009559,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,3.114684,0,0,1.284407,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.622795,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.537178,1.957669,0,0,1.398138,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.943016,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,1.032057,0,0,1.341272,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.455515,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,1.726266,0,0,1.170675,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.252119,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,1.726266,0,0,1.284407,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.092858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.537178,0.800654,0,0,1.170675,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.821484,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,1.494863,0,0,1.284407,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.729982,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.506521,1.263460,0,0,1.568735,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.651693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,0.569251,0,0,1.170675,0,0.918558];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.559166,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.264185,0.800654,0,0,1.056944,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.434760,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0,0.124958,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.687324,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0.800654,0,0,1.227541,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.100075,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.779513,1.032057,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.408954,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.779513,0.800654,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.234417,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.385353,0.337848,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.263719,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.264185,0.106445,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.911148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.537178,0.800654,0,0,1.284407,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.672269,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.779513,2.420475,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.928845,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,2.651878,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369127,-0.401824792392194,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.722137,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.143017,0.106445,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.978319,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0.337848,0,0,1.455004,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.127678,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,1.032057,0,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.159182,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.143017,0.106445,0,0,1.284407,0,1.444450];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.978319,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.748856,0,0.124958,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.584929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,0.800654,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.298719,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,1.032057,0,0,1.398138,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.633548,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.627688,0,1.744778,0,1.398138,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.952109,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.748856,0.337848,0,0,1.398138,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.797819,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,1.032057,0,0,1.227541,0,0.787085];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.068532,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,1.726266,0,0,1.056944,0,1.050031];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.733922,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.264185,0,0.124958,0,1.341272,0,1.181504];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.911919,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,1.726266,0,0,1.227541,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.167465,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.506521,0.337848,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.498618,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.658345,1.494863,0,0,1.284407,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.750592,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0.569251,0,0,1.341272,0,1.312977];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.579628999369126,-0.401824792392193,-1.148762873699773];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.245349,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.401508,0,0.337848,0,0.535296,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.474765951003068, 0.602503019232043,-0.094367923327578];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.539351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0.337848,0,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.510249075084806, 0.602503019232043,-0.05888479924584 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.830086,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.280340,0,0.106445,0,0.649027,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.245573631759551, 0.784910024735424, 0.676439757428905];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.431985,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,0,1.744778,0.137236,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.002826169873429, 2.180630376850361, 1.433692295542782];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.431250,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.795669,0,0,0.587764,0.478430,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.549592343591524, 1.528114879760256, 0.980458469260878];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.258035,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,0,0.587764,0.421564,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.130139864159424, 1.307944071136357, 0.561005989828778];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.316228,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0.569251,0,0.535296,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.835992387618052, 0.30436253243045 , 0.266858513287406];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.185525,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.143017,0,1.513375,0,0.260824,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.978944366758487, 1.15674857373542 , 0.409810492427841];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.178409,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.916837,0,0,0.356361,0.478430,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.955602603016623, 1.133406809993556, 0.386468728685977];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.884854,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.779513,0,0.819166,0.080370,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.151821575375336, 1.329625782352269, 0.582687701044689];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.943749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0,2.438987,0,0.147093,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.018287635265205, 2.196091842242137, 1.449153760934558];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.869122,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.068662,0,0,0.124958,0.250967,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.759682761966308, 0.937486968943241, 0.190548887635661];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "0.840759,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.189830,0,0,1.976181,0.137236,0,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.83981022494736 , 2.017614431924293, 1.270676350616714];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.564440,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.310998,0,0,0.356361,0.535296,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.168974846399228, 1.296711914628203, 0.599840972068582];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.467353,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.294842,0,0.356361,0,0.090227,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.3869455631241  , 0.564749770101033,-0.182188311206547];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "0.874655,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0.106445,0,0.364699,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.510538234793385, 0.688342441770318,-0.058595639537261];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.773848,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.294842,0,0.124958,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.973055005348913, 1.06530894949615 , 0.403921131018266];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.237536,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,0,0.819166,0.194102,0,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.739661103276346, 0.917465310253279, 0.1705272289457  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.326967,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.432165,0,0,1.976181,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.824278726405337, 2.916532670552574, 2.255144852074691];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.226083,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.294842,0,1.281972,0.080370,0,0,0.129720];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.220208595581322, 1.398012802558255, 0.651074721250676];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.786904,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.068662,0,0.337848,0,0.592162,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 1 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.026830284834102, 0.215953853534502, 1.457696410503456];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.789262,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.310998,0,0,0.587764,0.137236,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.845811344213163, 1.023615551190096, 0.276677469882516];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.169223,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0,1.281972,0.649027,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.633991422551871, 2.173327815527743, 2.064857548221224];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.880505,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.310998,0,0,0.587764,0.535296,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.112398302118555, 1.290202509095488, 0.543264427787909];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.435810,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0,0.356361,0.307833,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.785005491048866, 0.962809698025799, 0.21587161671822 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.471921,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.916837,0,0,0.124958,0.364699,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.741941199925439, 0.919745406902372, 0.172807325594792];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.671192,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.159173,0,0,0.587764,0.592162,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.514109219509786, 1.528114879760256, 0.94497534517914 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.603797,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0,0.124958,0.705893,0,0.659118,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 1 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.436411528975603, 0.696501345839478, 1.867277654644956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.527973,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.189830,0,0,0.356361,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.204457970480966, 1.296711914628203, 0.63532409615032 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.363927,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,0,1.050569,0,0.147093,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.629869844472885, 0.807674051449818, 0.060735970142239];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.632529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,0,1.513375,0.023504,0,0,0.129720];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.394745856724123, 1.572550063701056, 0.825611982393476];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.082610,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,0,1.513375,0,0.033361,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.206407182715497, 1.38421138969243 , 0.63727330838485 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.223939,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,0,0.819166,0.080370,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.888875635336589, 1.066679842313522, 0.319741761005942];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "0.953796,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.189830,0,0,0.819166,0.762759,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 1 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.906798626281399, 1.465317507265758, 2.337664751950752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.299400,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.537178,0,0.124958,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.973055005348913, 1.06530894949615 , 0.403921131018266];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.479871,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.189830,0,0.800654,0,0.421564,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.310389084859448, 0.139697088967937,-0.258744789471198];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.939693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0.106445,0,0.535296,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.969114856173869, 0.833905984364096, 0.399980981843223];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.075918,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0,1.744778,0.364699,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.230288985830438, 2.408093192807371, 1.661155111499792];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.796036,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.294842,0,0.124958,0.194102,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.439871117938308, 0.617675324915241,-0.129262756392339];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "0.747127,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,0,1.281972,0.137236,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.540020239609322, 1.717824446586255, 0.970886365278675];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.968444,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.416010,0,1.050569,0.364699,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.404607120414904, 1.582411327391837, 0.835473246084257];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.833031,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.310998,0,0,0.124958,0.478430,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.855672607903943, 1.033476814880876, 0.286538733573297];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.179084,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,0,1.050569,0.137236,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.177144304457894, 1.354948511434827, 0.608010430127248];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.242064,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.021849,0,1.744778,0,0.260824,0,0.261193];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.210347331890541, 1.388151538867474, 0.641213457559894];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.288644,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.294842,0,0.819166,0.250967,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.190945717323719, 1.368749924300652, 0.621811842993073];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.498202,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,0,0.124958,0.250967,0,0.001753,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.365263851908187, 0.54306805888512 ,-0.20387002242246 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "0.551352,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,0,0.356361,0.250967,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.728139787059614, 0.905943994036547, 0.159005912728968];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "0.914228,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.432165,0,0,0.356361,0.307833,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.785005491048866, 0.962809698025799, 0.21587161671822 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.229585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.900681,0,1.281972,0,0.431421,0,0.129720];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.708417259678051, 0.886221466654984, 0.139283385347404];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.743795,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,0,0.587764,0.194102,0,0.133226,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.902677048202415, 1.080481255179348, 0.333543173871768];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.088765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0.569251,0,1.274550,0,1.710902,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["100: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.603525033721982,-1.618148349604194, 6.034391159391336];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p100=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p100.svg"}], p100];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.108036,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,0,0.819166,0.762759,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["101: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.09005535645576 , 1.070898597207637, 3.520921482125114];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p101=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p101.svg"}], p101];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.556495,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.522676,0,0,0.124958,1.217684,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["102: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.549461176917902,-0.341181870141289, 4.980327302587256];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p102.svg"}], p102];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.050328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0,0.356361,1.047087,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["103: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.085816075972319, 0.455237117016644, 3.516682201641673];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p103.svg"}], p103];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.747372,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.795669,0,0,0.124958,1.160819,0,1.316483,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["104: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.773282975008264,-0.41578913617141 , 5.204149100677618];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p104=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p104.svg"}], p104];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.397928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[2.128516,0,0,0.124958,1.615744,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["105: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.7436409606922  ,-0.739241798066054, 6.174507086361554];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p105=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p105.svg"}], p105];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.054227,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,1.143017,0,1.281972,0.421564,0,0.659118,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["106: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.740440794797084, 2.137844691446006, 2.171306920466437];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p106=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p106.svg"}], p106];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.505672,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.765012,0,0,0.356361,1.445147,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["107: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.279995859746619, 0.057177189091877, 4.710861985415973];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p107=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p107.svg"}], p107];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.357110,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0,1.281972,1.160819,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["108: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.352622160436047, 1.267117569566351, 4.783488286105401];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p108=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p108.svg"}], p108];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.271387,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.643844,0,1.263460,0,1.331416,0,1.710902,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["109: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.07991325029358 ,-2.369222948989608, 5.510779375962934];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p109=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p109.svg"}], p109];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.949622,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.795669,0,0.337848,0,0.762759,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["110: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.327459440853615,-0.217589198472002, 2.758325566522969];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p110=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p110.svg"}], p110];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.949813,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0,0.819166,0.876490,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["111: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.431249580391274, 0.957167189229133, 3.862115706060628];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p111=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p111.svg"}], p111];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.292221,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.159173,0,0,0.124958,0.990221,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["112: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.867072729046872,-0.113719054184278, 4.297938854716226];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p112=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p112.svg"}], p112];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.459361,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.173674,0,1.281972,0.705893,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["113: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.776683084810231, 1.459097261441623, 4.207549210479585];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p113=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p113.svg"}], p113];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.215076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,0,0.587764,0.762759,0,1.579429,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["114: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.830746941614312, 0.182130781978717, 5.261613067283666];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p114=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p114.svg"}], p114];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.982457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0.337848,0,0.876490,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["115: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.851910394963492,-0.725739516508627, 4.282776520632846];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p115=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p115.svg"}], p115];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.336795,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.795669,0,0,0.124958,0.990221,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["116: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.683815998872509, 0.280699855873842, 3.114682124541863];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p116=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p116.svg"}], p116];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.701439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[2.249683,0,1.726266,0,1.672610,0,1.316483,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["117: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.457433261661653,-2.778804193131106, 4.888299387331007];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p117.svg"}], p117];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "6.965042,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[2.249683,0,0,1.050569,1.786341,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["118: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-8.969881977239927,-0.247172989544347, 8.40074810290928 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p118.svg"}], p118];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "6.534550,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.189830,0,0,1.976181,0.705893,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["119: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.498797429915785, 2.810671006934649, 2.929663555585139];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p119=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p119.svg"}], p119];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.268076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.280340,0,0.337848,0,1.103953,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["120: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.534298842834521,-0.953202332465637, 4.965164968503875];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p120.svg"}], p120];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.170097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.294842,0,0.587764,0.649027,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["121: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.911877077446316, 0.821754070034716, 3.34274320311567 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p121=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p121.svg"}], p121];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.585169,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[2.249683,0,0,0.587764,1.672610,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["122: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.982625092865944,-0.201828601771826, 6.413491218535298];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p122=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p122.svg"}], p122];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.563594,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0,0.819166,0.649027,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["123: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.354442222462127, 1.316102975205515, 2.785308348131481];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p123=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p123.svg"}], p123];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.812118,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0.569251,0,1.103953,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["124: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.514058057586229,-0.921659357558943, 3.944924183255583];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p124=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p124.svg"}], p124];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.896219,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.643844,0,0.337848,0,1.274550,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["125: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.073995628447189,-0.466434594336526, 3.504861754116543];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p125=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p125.svg"}], p125];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.046833,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.432165,0,0,0.587764,0.592162,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["126: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.952442145362316, 1.141565714062716, 2.38330827103167 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p126=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p126.svg"}], p126];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.402681,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.310998,0,0,0.124958,0.649027,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["127: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.660233327065968, 0.621894079809356, 2.091099452735321];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p127=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p127.svg"}], p127];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.875573,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0,0.587764,1.047087,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["128: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.500475771278736, 0.292221172090576, 4.93134189694809 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p128=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p128.svg"}], p128];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.494362,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.643844,0,0,0.124958,1.160819,0,0.527645,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["129: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.40676951465954 , 0.373048683944832, 2.837635640328893];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p129=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p129.svg"}], p129];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.457265,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.886180,0,0,0.587764,1.331416,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["130: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.564623511001279, 0.270838592183061, 4.995489636670633];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p130=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p130.svg"}], p130];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.727423,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[2.492019,0,1.726266,0,1.502013,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["131: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.156804105642141,-2.345261141124602, 3.587670231311495];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p131=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p131.svg"}], p131];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "6.773835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0,0.587764,1.047087,0,1.316483,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["132: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.894894681336857, 0.160748202071202, 5.325760807006211];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p132=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p132.svg"}], p132];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.625835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0,0.587764,0.762759,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["133: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.280976751091225, 1.365387512153079, 1.711842876760578];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p133=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p133.svg"}], p133];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.300027,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.310998,0,0,1.050569,1.047087,0,0.264699,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["134: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.202349331135994, 1.675337892490297, 2.633215456805348];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p134.svg"}], p134];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.673353,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[2.249683,0,0,0.124958,1.331416,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["135: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.679493220969657,-0.717859218158541, 6.110359346639011];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p135=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p135.svg"}], p135];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "5.154012,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0.800654,0,1.047087,0,1.579429,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["136: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.295314710660779,-1.490615528759864, 4.726180836330133];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p136.svg"}], p136];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.980503,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.674501,0,0.106445,0,0.990221,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["137: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.452413033740457, 0.049296890741789, 2.883279159409811];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p137.svg"}], p137];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.561759,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.189830,0,0,0.124958,0.592162,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["138: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.489636215098209, 0.678759783798609, 1.920502340767563];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p138.svg"}], p138];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.697540,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.280340,0,0.106445,0,0.933356,0,1.185010,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["139: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.465072651947063,-0.28825631532708 , 3.895938777616417];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p139=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p139.svg"}], p139];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.505151,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0.106445,0,1.047087,0,1.579429,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["140: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.989523606056937,-0.796406633363705, 5.420389731726291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p140=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p140.svg"}], p140];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.770966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.280340,0,0.106445,0,0.762759,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["141: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.742119136160031,-0.38060514339807 , 4.172985261829385];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p141=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p141.svg"}], p141];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.597500,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0,0.052506,0,0.819166,0.762759,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["142: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.09005535645576 , 1.070898597207637, 3.520921482125114];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p142=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p142.svg"}], p142];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.556495,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.159173,0,0.337848,0,1.217684,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["143: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.875493066770036,-1.066933740444142, 5.30635919243939 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p143=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p143.svg"}], p143];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.162661,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0.569251,0,1.103953,0,1.710902,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["144: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-6.091733697818712,-1.447551237636437, 5.522599823488066];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p144=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p144.svg"}], p144];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "4.422111,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[1.038005,0,0,0.124958,0.819624,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["145: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.144119213259843,-0.206067882255269, 4.574985338929197];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p145=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p145.svg"}], p145];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.430542,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.553333,0,0,1.281972,0.705893,0,0.922064,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["146: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.382264174752111, 1.590570231460996, 3.813130300421465];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p146=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p146.svg"}], p146];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.463262,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.795669,0,0,0.124958,0.819624,0,1.053537,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["147: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.960862483085481, 0.188351027802853, 3.391728608754835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p147=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p147.svg"}], p147];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "2.793788,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.432165,0,0.800654,0,0.933356,0,1.447956,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["148: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-4.559701576667145,-1.245411150761986, 3.990567702336499];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p148=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p148.svg"}], p148];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "3.614131,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

rsys=IRISReduced[0.068662,0,0,0.124958,0.762759,0,0.790591,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["149: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.001427551001481, 0.508162671830852, 2.432293676670835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p149=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/iris", "IRISReduced_p149.svg"}], p149];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["IRISReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["IRISReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["IRISReduced_convergence_times.csv", ","];
  ];
  WriteString["IRISReduced_convergence_times.csv", "1.746969,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["IRISReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["IRISReduced_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
