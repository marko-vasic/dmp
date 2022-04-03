#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

HEART[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x1Spos, x1SposInitial],
rxn[x0Spos,iL0N0Spos+iL0N2Spos+iL0N3Spos+iL0N4Sneg,1],
rxn[x1Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Sneg+iL0N4Sneg,1],
conc[iL0N0Sneg,2.3060997487],
conc[iL0N1Sneg,2.1444769959],
conc[iL0N2Sneg,4.0078084010],
conc[iL0N3Sneg,2.3041906101],
conc[iL0N4Spos,1.0908678161],
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
rxn[hL0N0Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N0Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N1Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N1Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N2Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N2Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N3Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N3Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N4Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N4Sneg,hL1N0Spos+hL1N1Sneg,1],
conc[hL1N0Spos,0.9311815160],
conc[hL1N1Sneg,0.9009135303]
}

HEARTReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_] =
  reduceFFNCCrn[HEART[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], keepInputLayer=True, inputPrefix="x"];

cnt = Count[HEART[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[HEARTReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], rxn[___]];
Print["Number of reactions (reduced form): " <> ToString[cntReduced]];

tmax = 50;
totalCount = 0;
correctCount = 0;
matchingCount = 0;

analyzeConvergenceTime = True;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=HEART[3.250166,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.956859433588297, 0.987127419268529];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.251166,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.144567811426432, 0.174835797106664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.845020,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.667723333898275,-0.637455348218043];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.438874,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.032729,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.626583,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.220437,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654605236764822,-0.62433725108459 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.814292,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.248459485288534,-0.218191499608302];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.408146,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.157686206583109, 0.187954192263341];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.002000,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.248459485288534,-0.218191499608302];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.408146,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654605236764822,-0.62433725108459 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.814292,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.220437,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.626583,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.032729,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.667723333898275,-0.637455348218043];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.438874,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.144567811426432, 0.174835797106664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.845020,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.956859433588297, 0.987127419268529];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.251166,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.493949735101969, 0.524217720782201];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.482621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.318341887059896,-0.288073901379664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.076475,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.670329,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.264184,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.858038,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.451892,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.886060086007986,-0.855792100327754];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.045746,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.479914334531698,-0.449646348851466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.639601,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.073768642660055,-0.043500656979823];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.233455,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.479914334531698,-0.449646348851466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.639601,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.886060086007986,-0.855792100327754];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.045746,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.451892,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.858038,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.264184,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.670329,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.318341887059896,-0.288073901379664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.076475,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.493949735101969, 0.524217720782201];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.482621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.031039798197062, 0.061307783877294];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.714076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.781251823964804,-0.750983838284572];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.307930,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.901784,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.495638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.089493,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.683347,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.277201,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.711369302984152,-0.68110131730392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.871056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.305223611112509,-0.274955625432277];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.464910,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.711369302984152,-0.68110131730392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.871056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.277201,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.683347,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.089493,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.495638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.901784,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.781251823964804,-0.750983838284572];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.307930,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.031039798197062, 0.061307783877294];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.714076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.431870138707846,-0.401602153027614];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.945530,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.539385,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.133239,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.727093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.320948,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.914802,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.508656,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.102511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.536678579564962,-0.506410593884731];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.696365,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.102511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.508656,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.914802,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.320948,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.727093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.133239,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.539385,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.431870138707846,-0.401602153027614];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.945530,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.72560291474239 ,-0.695334929062158];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.176985,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.770840,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.364694,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.958548,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.552403,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.146257,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.740111,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.333966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.768133488412771,-0.737865502732539];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "0.927820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.333966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.740111,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.146257,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.552403,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.958548,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.364694,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.770840,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.72560291474239 ,-0.695334929062158];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.176985,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.530549565517584,-0.500281579837352];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.408441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.002295,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.596149,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.190003,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.783858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.377712,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.971566,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.565420,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.159275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.565420,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.971566,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.377712,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.783858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.190003,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.596149,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["100: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p100=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p100.svg"}], p100];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.002295,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["101: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.530549565517584,-0.500281579837352];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p101=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p101.svg"}], p101];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.408441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["102: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.29909471627442 ,-0.268826730594188];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p102.svg"}], p102];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.639895,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["103: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.705240527355352,-0.67497254167512 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p103.svg"}], p103];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.233749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["104: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p104=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p104.svg"}], p104];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.827604,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["105: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p105=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p105.svg"}], p105];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.421458,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["106: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p106=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p106.svg"}], p106];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.015312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["107: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p107=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p107.svg"}], p107];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.609167,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["108: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p108=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p108.svg"}], p108];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.203021,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["109: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p109=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p109.svg"}], p109];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.796875,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["110: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p110=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p110.svg"}], p110];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.390730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["111: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p111=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p111.svg"}], p111];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.796875,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["112: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p112=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p112.svg"}], p112];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.203021,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["113: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p113=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p113.svg"}], p113];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.609167,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["114: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p114=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p114.svg"}], p114];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.015312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["115: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p115=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p115.svg"}], p115];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.421458,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["116: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p116=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p116.svg"}], p116];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.827604,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["117: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.705240527355352,-0.67497254167512 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p117.svg"}], p117];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.233749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["118: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.29909471627442 ,-0.268826730594188];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p118.svg"}], p118];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.639895,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["119: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.067639747821966,-0.037371762141734];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p119=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p119.svg"}], p119];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.871350,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["120: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.473785558902899,-0.443517573222667];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p120.svg"}], p120];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.465204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["121: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.879931131565252,-0.84966314588502 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p121=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p121.svg"}], p121];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.059059,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["122: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p122=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p122.svg"}], p122];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.652913,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["123: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p123=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p123.svg"}], p123];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.246768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["124: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p124=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p124.svg"}], p124];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.840621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["125: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p125=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p125.svg"}], p125];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.434476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["126: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p126=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p126.svg"}], p126];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.028330,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["127: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p127=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p127.svg"}], p127];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.622185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["128: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p128=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p128.svg"}], p128];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.028330,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["129: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p129=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p129.svg"}], p129];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.434476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["130: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p130=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p130.svg"}], p130];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.840621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["131: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p131=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p131.svg"}], p131];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.246768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["132: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p132=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p132.svg"}], p132];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.652913,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["133: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.879931131565252,-0.84966314588502 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p133=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p133.svg"}], p133];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.059059,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["134: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.473785558902899,-0.443517573222667];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p134.svg"}], p134];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.465204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["135: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.067639747821966,-0.037371762141734];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p135=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p135.svg"}], p135];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.871350,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["136: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.163815220630487, 0.194083206310719];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p136.svg"}], p136];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.102805,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["137: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.242330590450445,-0.212062604770213];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p137.svg"}], p137];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.696659,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["138: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.648476163112799,-0.618208177432567];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p138.svg"}], p138];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.290514,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["139: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p139=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p139.svg"}], p139];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.884368,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["140: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p140=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p140.svg"}], p140];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.478222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["141: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p141=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p141.svg"}], p141];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.072077,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["142: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p142=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p142.svg"}], p142];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.665931,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["143: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p143=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p143.svg"}], p143];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.259785,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["144: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p144=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p144.svg"}], p144];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "1.853639,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["145: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p145=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p145.svg"}], p145];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.259785,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["146: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p146=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p146.svg"}], p146];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.665931,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["147: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p147=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p147.svg"}], p147];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.072077,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["148: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p148=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p148.svg"}], p148];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.478222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["149: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p149=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p149.svg"}], p149];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.884368,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["150: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.648476163112799,-0.618208177432567];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p150=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p150.svg"}], p150];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.290514,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["151: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.242330590450445,-0.212062604770213];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p151=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p151.svg"}], p151];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.696659,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["152: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.163815220630487, 0.194083206310719];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p152=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p152.svg"}], p152];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.102805,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["153: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.395270069873652, 0.425538055553883];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p153=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p153.svg"}], p153];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.334260,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["154: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.010875741207281,0.019392244472951];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p154=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p154.svg"}], p154];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.928114,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["155: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.417021313869635,-0.386753328189403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p155=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p155.svg"}], p155];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.521969,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["156: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.823167124950567,-0.792899139270335];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p156=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p156.svg"}], p156];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.115823,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["157: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p157=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p157.svg"}], p157];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.709677,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["158: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p158=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p158.svg"}], p158];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.303531,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["159: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p159=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p159.svg"}], p159];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.897386,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["160: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p160=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p160.svg"}], p160];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.491240,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["161: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p161=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p161.svg"}], p161];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.085094,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["162: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p162=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p162.svg"}], p162];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.491240,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["163: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p163=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p163.svg"}], p163];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.897386,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["164: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p164=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p164.svg"}], p164];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.303531,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["165: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p165=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p165.svg"}], p165];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.709677,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["166: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.823167124950567,-0.792899139270335];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p166=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p166.svg"}], p166];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.115823,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["167: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.417021313869635,-0.386753328189403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p167=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p167.svg"}], p167];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.521969,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["168: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.010875741207281,0.019392244472951];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p168=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p168.svg"}], p168];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.928114,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["169: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.395270069873652, 0.425538055553883];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p169=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p169.svg"}], p169];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.334260,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["170: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.797797535279796, 0.828065520960028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p170=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p170.svg"}], p170];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.565715,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["171: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.391651724198864, 0.421919709879095];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p171=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p171.svg"}], p171];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.159569,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["172: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.01449384846349 ,0.015774137216742];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p172=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p172.svg"}], p172];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.753424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["173: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.420639659544423,-0.390371673864191];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p173=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p173.svg"}], p173];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.347278,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["174: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p174=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p174.svg"}], p174];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.941132,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["175: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p175=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p175.svg"}], p175];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.534986,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["176: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p176=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p176.svg"}], p176];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.128841,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["177: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p177=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p177.svg"}], p177];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.722695,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["178: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p178=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p178.svg"}], p178];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.316549,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["179: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p179=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p179.svg"}], p179];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.722695,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["180: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p180=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p180.svg"}], p180];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.128841,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["181: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p181=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p181.svg"}], p181];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.534986,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["182: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p182=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p182.svg"}], p182];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.941132,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["183: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.420639659544423,-0.390371673864191];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p183=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p183.svg"}], p183];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.347278,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["184: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.01449384846349 ,0.015774137216742];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p184=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p184.svg"}], p184];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.753424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["185: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.391651724198864, 0.421919709879095];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p185=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p185.svg"}], p185];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.159569,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["186: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.797797535279796, 0.828065520960028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p186=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p186.svg"}], p186];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.565715,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["187: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.260707233766124, 1.290975219446356];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p187=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p187.svg"}], p187];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.797170,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["188: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.854561422685192, 0.884829408365424];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p188=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p188.svg"}], p188];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.391024,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["189: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.448415850022838, 0.47868383570307 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p189=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p189.svg"}], p189];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.984879,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["190: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042270038941905, 0.072538024622137];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p190=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p190.svg"}], p190];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.578732,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["191: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.363875533720448,-0.333607548040216];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p191=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p191.svg"}], p191];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.172587,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["192: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p192=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p192.svg"}], p192];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.766441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["193: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p193=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p193.svg"}], p193];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.360296,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["194: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p194=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p194.svg"}], p194];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.954150,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["195: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p195=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p195.svg"}], p195];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.548004,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["196: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p196=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p196.svg"}], p196];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.954150,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["197: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p197=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p197.svg"}], p197];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.360296,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["198: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p198=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p198.svg"}], p198];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.766441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["199: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.363875533720448,-0.333607548040216];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p199=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p199.svg"}], p199];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.172587,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["200: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042270038941905, 0.072538024622137];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p200=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p200.svg"}], p200];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.578732,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["201: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.448415850022838, 0.47868383570307 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p201=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p201.svg"}], p201];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.984879,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["202: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.854561422685192, 0.884829408365424];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p202=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p202.svg"}], p202];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.391024,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["203: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.260707233766124, 1.290975219446356];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p203=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p203.svg"}], p203];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.797170,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["204: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.723616932252452, 1.753884917932684];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p204=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p204.svg"}], p204];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.028625,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["205: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.31747112117152 , 1.347739106851752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p205=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p205.svg"}], p205];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.622479,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["206: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.911325548509166, 0.941593534189398];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p206=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p206.svg"}], p206];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.216333,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["207: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.505179737428234, 0.535447723108466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p207=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p207.svg"}], p207];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.810187,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["208: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.09903416476588 , 0.129302150446112];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p208=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p208.svg"}], p208];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.404042,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["209: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p209=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p209.svg"}], p209];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.997896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["210: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p210=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p210.svg"}], p210];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.591751,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["211: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p211=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p211.svg"}], p211];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.185605,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["212: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p212=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p212.svg"}], p212];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "2.779459,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["213: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p213=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p213.svg"}], p213];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.185605,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["214: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p214=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p214.svg"}], p214];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.591751,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["215: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p215=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p215.svg"}], p215];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.997896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["216: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.09903416476588 , 0.129302150446112];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p216=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p216.svg"}], p216];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.404042,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["217: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.505179737428234, 0.535447723108466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p217=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p217.svg"}], p217];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.810187,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["218: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.911325548509166, 0.941593534189398];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p218=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p218.svg"}], p218];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.216333,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["219: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.31747112117152 , 1.347739106851752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p219=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p219.svg"}], p219];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.622479,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["220: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.723616932252452, 1.753884917932684];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p220=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p220.svg"}], p220];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.028625,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["221: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.186526630738781, 2.216794616419012];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p221=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p221.svg"}], p221];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.260079,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["222: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.780380819657848, 1.81064880533808 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p222=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p222.svg"}], p222];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.853934,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["223: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.374235246995494, 1.404503232675726];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p223=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p223.svg"}], p223];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.447788,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["224: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.968089435914562, 0.998357421594794];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p224=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p224.svg"}], p224];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.041642,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["225: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.561943863252208, 0.59221184893244 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p225=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p225.svg"}], p225];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.635497,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["226: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155798052171276, 0.186066037851508];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p226=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p226.svg"}], p226];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.229351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["227: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p227=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p227.svg"}], p227];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.823205,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["228: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p228=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p228.svg"}], p228];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.417060,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["229: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p229=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p229.svg"}], p229];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.010914,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["230: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p230=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p230.svg"}], p230];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.417060,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["231: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p231=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p231.svg"}], p231];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.823205,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["232: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155798052171276, 0.186066037851508];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p232=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p232.svg"}], p232];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.229351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["233: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.561943863252208, 0.59221184893244 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p233=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p233.svg"}], p233];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.635497,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["234: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.968089435914562, 0.998357421594794];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p234=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p234.svg"}], p234];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.041642,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["235: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.374235246995494, 1.404503232675726];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p235=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p235.svg"}], p235];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.447788,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["236: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.780380819657848, 1.81064880533808 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p236=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p236.svg"}], p236];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.853934,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["237: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.186526630738781, 2.216794616419012];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p237=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p237.svg"}], p237];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.260079,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["238: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.649436329225109, 2.679704314905341];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p238=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p238.svg"}], p238];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.491534,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["239: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.243290518144176, 2.273558503824408];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p239=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p239.svg"}], p239];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.085389,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["240: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.837144945481822, 1.867412931162054];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p240=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p240.svg"}], p240];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.679243,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["241: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.430999134400889, 1.461267120081121];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p241=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p241.svg"}], p241];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.273097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["242: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.024853561738536, 1.055121547418768];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p242=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p242.svg"}], p242];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.866952,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["243: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.618707750657604, 0.648975736337835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p243=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p243.svg"}], p243];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.460806,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["244: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.212562118390605, 0.242830104070837];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p244=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p244.svg"}], p244];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.054660,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["245: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.165710258690592, 0.195978244370824];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p245=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p245.svg"}], p245];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.648515,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.001000,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["246: 0 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.165710258690592, 0.195978244370824];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p246=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p246.svg"}], p246];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.242369,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.407146,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["247: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.165710258690592, 0.195978244370824];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p247=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p247.svg"}], p247];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "3.648515,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.813291,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["248: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.212562118390605, 0.242830104070837];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p248=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p248.svg"}], p248];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.054660,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.219437,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["249: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.618707750657604, 0.648975736337835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p249=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p249.svg"}], p249];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.460806,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.625583,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["250: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.024853561738536, 1.055121547418768];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p250=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p250.svg"}], p250];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "4.866952,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.031729,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["251: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.430999134400889, 1.461267120081121];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p251=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p251.svg"}], p251];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.273097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.437874,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["252: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.837144945481822, 1.867412931162054];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p252=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p252.svg"}], p252];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "5.679243,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[2.844020,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["253: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.243290518144176, 2.273558503824408];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p253=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p253.svg"}], p253];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.085389,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[3.250166,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["254: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.649436329225109, 2.679704314905341];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p254=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEART_p254.svg"}], p254];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEART_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEART_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEART_convergence_times.csv", ","];
  ];
  WriteString["HEART_convergence_times.csv", "6.491534,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
tmax = 50;
totalCount = 0;
correctCount = 0;
matchingCount = 0;

analyzeConvergenceTime = True;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=HEARTReduced[3.250166,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.956859433588297, 0.987127419268529];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.251166,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.144567811426432, 0.174835797106664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.845020,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.667723333898275,-0.637455348218043];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.438874,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.032729,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.626583,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.220437,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654605236764822,-0.62433725108459 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.814292,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.248459485288534,-0.218191499608302];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.408146,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.157686206583109, 0.187954192263341];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.002000,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.248459485288534,-0.218191499608302];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.408146,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.654605236764822,-0.62433725108459 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.814292,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.220437,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.626583,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.032729,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.667723333898275,-0.637455348218043];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.438874,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.144567811426432, 0.174835797106664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.845020,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.001000,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.956859433588297, 0.987127419268529];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.251166,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.493949735101969, 0.524217720782201];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.482621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.318341887059896,-0.288073901379664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.076475,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.670329,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.264184,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.858038,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.451892,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.886060086007986,-0.855792100327754];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.045746,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.479914334531698,-0.449646348851466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.639601,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 1 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.073768642660055,-0.043500656979823];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.233455,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.479914334531698,-0.449646348851466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.639601,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.886060086007986,-0.855792100327754];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.045746,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.451892,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.858038,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.264184,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.670329,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.318341887059896,-0.288073901379664];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.076475,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.232455,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.493949735101969, 0.524217720782201];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.482621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.031039798197062, 0.061307783877294];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.714076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.781251823964804,-0.750983838284572];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.307930,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.901784,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.495638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.089493,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.683347,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.277201,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.711369302984152,-0.68110131730392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.871056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.305223611112509,-0.274955625432277];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.464910,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.711369302984152,-0.68110131730392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.871056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.277201,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.683347,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.089493,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.495638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.901784,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.781251823964804,-0.750983838284572];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.307930,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.463910,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.031039798197062, 0.061307783877294];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.714076,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.431870138707846,-0.401602153027614];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.945530,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.539385,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.133239,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.727093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.320948,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.914802,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.508656,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.102511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.536678579564962,-0.506410593884731];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.696365,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.102511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.508656,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.914802,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.320948,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.727093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.133239,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.539385,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.695365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.431870138707846,-0.401602153027614];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.945530,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.72560291474239 ,-0.695334929062158];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.176985,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.770840,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.364694,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.958548,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.552403,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.146257,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.740111,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.333966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.768133488412771,-0.737865502732539];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "0.927820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.333966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.740111,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.146257,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.552403,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.958548,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.364694,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.770840,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,0.926820,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.72560291474239 ,-0.695334929062158];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.176985,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.530549565517584,-0.500281579837352];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.408441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.002295,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.596149,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.190003,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.783858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.377712,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.971566,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.565420,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.159275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.565420,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.971566,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.377712,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.783858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.190003,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.596149,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["100: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p100=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p100.svg"}], p100];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.002295,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.158275,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["101: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.530549565517584,-0.500281579837352];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p101=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p101.svg"}], p101];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.408441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["102: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.29909471627442 ,-0.268826730594188];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p102.svg"}], p102];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.639895,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["103: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.705240527355352,-0.67497254167512 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p103.svg"}], p103];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.233749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["104: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p104=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p104.svg"}], p104];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.827604,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["105: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p105=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p105.svg"}], p105];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.421458,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["106: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p106=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p106.svg"}], p106];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.015312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["107: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p107=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p107.svg"}], p107];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.609167,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["108: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p108=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p108.svg"}], p108];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.203021,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["109: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p109=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p109.svg"}], p109];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.796875,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["110: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p110=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p110.svg"}], p110];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.390730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["111: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p111=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p111.svg"}], p111];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.796875,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["112: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p112=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p112.svg"}], p112];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.203021,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["113: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p113=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p113.svg"}], p113];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.609167,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["114: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p114=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p114.svg"}], p114];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.015312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["115: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p115=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p115.svg"}], p115];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.421458,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["116: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p116=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p116.svg"}], p116];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.827604,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["117: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.705240527355352,-0.67497254167512 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p117.svg"}], p117];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.233749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.389729,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["118: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.29909471627442 ,-0.268826730594188];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p118.svg"}], p118];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.639895,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["119: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.067639747821966,-0.037371762141734];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p119=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p119.svg"}], p119];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.871350,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["120: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.473785558902899,-0.443517573222667];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p120.svg"}], p120];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.465204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["121: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.879931131565252,-0.84966314588502 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p121=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p121.svg"}], p121];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.059059,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["122: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p122=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p122.svg"}], p122];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.652913,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["123: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p123=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p123.svg"}], p123];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.246768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["124: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p124=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p124.svg"}], p124];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.840621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["125: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p125=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p125.svg"}], p125];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.434476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["126: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p126=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p126.svg"}], p126];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.028330,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["127: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p127=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p127.svg"}], p127];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.622185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["128: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p128=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p128.svg"}], p128];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.028330,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["129: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p129=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p129.svg"}], p129];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.434476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["130: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p130=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p130.svg"}], p130];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.840621,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["131: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p131=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p131.svg"}], p131];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.246768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["132: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p132=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p132.svg"}], p132];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.652913,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["133: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.879931131565252,-0.84966314588502 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p133=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p133.svg"}], p133];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.059059,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["134: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.473785558902899,-0.443517573222667];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p134.svg"}], p134];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.465204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.621184,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["135: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.067639747821966,-0.037371762141734];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p135=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p135.svg"}], p135];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.871350,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["136: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.163815220630487, 0.194083206310719];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p136.svg"}], p136];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.102805,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["137: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.242330590450445,-0.212062604770213];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p137.svg"}], p137];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.696659,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["138: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.648476163112799,-0.618208177432567];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p138.svg"}], p138];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.290514,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["139: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p139=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p139.svg"}], p139];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.884368,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["140: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p140=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p140.svg"}], p140];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.478222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["141: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p141=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p141.svg"}], p141];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.072077,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["142: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p142=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p142.svg"}], p142];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.665931,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["143: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p143=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p143.svg"}], p143];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.259785,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["144: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p144=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p144.svg"}], p144];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "1.853639,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["145: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p145=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p145.svg"}], p145];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.259785,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["146: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p146=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p146.svg"}], p146];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.665931,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["147: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p147=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p147.svg"}], p147];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.072077,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["148: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p148=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p148.svg"}], p148];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.478222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["149: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p149=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p149.svg"}], p149];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.884368,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["150: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.648476163112799,-0.618208177432567];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p150=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p150.svg"}], p150];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.290514,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["151: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.242330590450445,-0.212062604770213];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p151=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p151.svg"}], p151];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.696659,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,1.852639,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["152: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.163815220630487, 0.194083206310719];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p152=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p152.svg"}], p152];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.102805,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["153: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.395270069873652, 0.425538055553883];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p153=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p153.svg"}], p153];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.334260,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["154: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.010875741207281,0.019392244472951];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p154=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p154.svg"}], p154];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.928114,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["155: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.417021313869635,-0.386753328189403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p155=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p155.svg"}], p155];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.521969,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["156: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.823167124950567,-0.792899139270335];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p156=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p156.svg"}], p156];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.115823,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["157: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p157=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p157.svg"}], p157];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.709677,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["158: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p158=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p158.svg"}], p158];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.303531,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["159: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p159=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p159.svg"}], p159];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.897386,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["160: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p160=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p160.svg"}], p160];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.491240,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["161: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p161=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p161.svg"}], p161];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.085094,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["162: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p162=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p162.svg"}], p162];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.491240,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["163: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p163=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p163.svg"}], p163];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.897386,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["164: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p164=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p164.svg"}], p164];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.303531,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["165: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.931181516026466,-0.900913530346234];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p165=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p165.svg"}], p165];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.709677,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["166: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.823167124950567,-0.792899139270335];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p166=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p166.svg"}], p166];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.115823,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["167: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.417021313869635,-0.386753328189403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p167=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p167.svg"}], p167];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.521969,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["168: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.010875741207281,0.019392244472951];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p168=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p168.svg"}], p168];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.928114,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.084094,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["169: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.395270069873652, 0.425538055553883];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p169=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p169.svg"}], p169];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.334260,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["170: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.797797535279796, 0.828065520960028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p170=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p170.svg"}], p170];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.565715,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["171: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.391651724198864, 0.421919709879095];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p171=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p171.svg"}], p171];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.159569,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["172: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.01449384846349 ,0.015774137216742];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p172=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p172.svg"}], p172];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.753424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["173: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.420639659544423,-0.390371673864191];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p173=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p173.svg"}], p173];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.347278,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["174: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p174=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p174.svg"}], p174];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.941132,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["175: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p175=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p175.svg"}], p175];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.534986,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["176: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p176=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p176.svg"}], p176];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.128841,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["177: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p177=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p177.svg"}], p177];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.722695,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["178: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p178=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p178.svg"}], p178];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.316549,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["179: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p179=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p179.svg"}], p179];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.722695,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["180: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p180=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p180.svg"}], p180];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.128841,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["181: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p181=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p181.svg"}], p181];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.534986,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["182: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.760109138282064,-0.729841152601832];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p182=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p182.svg"}], p182];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.941132,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["183: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.420639659544423,-0.390371673864191];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p183=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p183.svg"}], p183];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.347278,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["184: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.01449384846349 ,0.015774137216742];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p184=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p184.svg"}], p184];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.753424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["185: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.391651724198864, 0.421919709879095];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p185=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p185.svg"}], p185];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.159569,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.315549,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["186: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.797797535279796, 0.828065520960028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p186=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p186.svg"}], p186];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.565715,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["187: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.260707233766124, 1.290975219446356];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p187=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p187.svg"}], p187];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.797170,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["188: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.854561422685192, 0.884829408365424];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p188=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p188.svg"}], p188];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.391024,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["189: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.448415850022838, 0.47868383570307 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p189=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p189.svg"}], p189];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.984879,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["190: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042270038941905, 0.072538024622137];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p190=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p190.svg"}], p190];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.578732,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["191: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.363875533720448,-0.333607548040216];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p191=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p191.svg"}], p191];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.172587,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["192: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p192=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p192.svg"}], p192];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.766441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["193: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p193=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p193.svg"}], p193];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.360296,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["194: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p194=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p194.svg"}], p194];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.954150,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["195: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p195=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p195.svg"}], p195];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.548004,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["196: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p196=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p196.svg"}], p196];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.954150,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["197: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p197=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p197.svg"}], p197];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.360296,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["198: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.5286542890389  ,-0.498386303358668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p198=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p198.svg"}], p198];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.766441,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["199: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.363875533720448,-0.333607548040216];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p199=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p199.svg"}], p199];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.172587,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["200: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.042270038941905, 0.072538024622137];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p200=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p200.svg"}], p200];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.578732,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["201: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.448415850022838, 0.47868383570307 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p201=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p201.svg"}], p201];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.984879,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["202: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.854561422685192, 0.884829408365424];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p202=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p202.svg"}], p202];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.391024,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.547004,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["203: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.260707233766124, 1.290975219446356];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p203=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p203.svg"}], p203];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.797170,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["204: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.723616932252452, 1.753884917932684];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p204=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p204.svg"}], p204];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.028625,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["205: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.31747112117152 , 1.347739106851752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p205=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p205.svg"}], p205];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.622479,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["206: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.911325548509166, 0.941593534189398];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p206=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p206.svg"}], p206];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.216333,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["207: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.505179737428234, 0.535447723108466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p207=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p207.svg"}], p207];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.810187,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["208: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.09903416476588 , 0.129302150446112];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p208=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p208.svg"}], p208];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.404042,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["209: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p209=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p209.svg"}], p209];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.997896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["210: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p210=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p210.svg"}], p210];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.591751,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["211: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p211=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p211.svg"}], p211];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.185605,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["212: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p212=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p212.svg"}], p212];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "2.779459,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["213: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p213=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p213.svg"}], p213];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.185605,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["214: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p214=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p214.svg"}], p214];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.591751,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["215: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.297199439795736,-0.266931454115504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p215=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p215.svg"}], p215];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.997896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["216: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.09903416476588 , 0.129302150446112];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p216=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p216.svg"}], p216];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.404042,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["217: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.505179737428234, 0.535447723108466];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p217=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p217.svg"}], p217];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.810187,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["218: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.911325548509166, 0.941593534189398];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p218=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p218.svg"}], p218];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.216333,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["219: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.31747112117152 , 1.347739106851752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p219=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p219.svg"}], p219];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.622479,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,2.778459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["220: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.723616932252452, 1.753884917932684];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p220=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p220.svg"}], p220];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.028625,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["221: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.186526630738781, 2.216794616419012];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p221=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p221.svg"}], p221];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.260079,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["222: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.780380819657848, 1.81064880533808 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p222=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p222.svg"}], p222];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.853934,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["223: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.374235246995494, 1.404503232675726];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p223=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p223.svg"}], p223];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.447788,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["224: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.968089435914562, 0.998357421594794];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p224=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p224.svg"}], p224];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.041642,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["225: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.561943863252208, 0.59221184893244 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p225=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p225.svg"}], p225];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.635497,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["226: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155798052171276, 0.186066037851508];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p226=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p226.svg"}], p226];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.229351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["227: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p227=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p227.svg"}], p227];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.823205,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["228: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p228=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p228.svg"}], p228];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.417060,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["229: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p229=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p229.svg"}], p229];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.010914,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["230: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p230=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p230.svg"}], p230];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.417060,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["231: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.065744590552572,-0.03547660487234 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p231=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p231.svg"}], p231];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.823205,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["232: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.155798052171276, 0.186066037851508];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p232=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p232.svg"}], p232];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.229351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["233: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.561943863252208, 0.59221184893244 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p233=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p233.svg"}], p233];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.635497,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["234: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.968089435914562, 0.998357421594794];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p234=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p234.svg"}], p234];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.041642,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["235: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.374235246995494, 1.404503232675726];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p235=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p235.svg"}], p235];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.447788,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["236: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.780380819657848, 1.81064880533808 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p236=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p236.svg"}], p236];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.853934,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,3.009914,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["237: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.186526630738781, 2.216794616419012];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p237=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p237.svg"}], p237];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.260079,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["238: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.649436329225109, 2.679704314905341];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p238=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p238.svg"}], p238];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.491534,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["239: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.243290518144176, 2.273558503824408];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p239=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p239.svg"}], p239];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.085389,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["240: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.837144945481822, 1.867412931162054];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p240=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p240.svg"}], p240];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.679243,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["241: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.430999134400889, 1.461267120081121];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p241=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p241.svg"}], p241];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.273097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["242: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.024853561738536, 1.055121547418768];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p242=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p242.svg"}], p242];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.866952,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["243: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.618707750657604, 0.648975736337835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p243=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p243.svg"}], p243];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.460806,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["244: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.212562118390605, 0.242830104070837];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p244=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p244.svg"}], p244];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.054660,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["245: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.165710258690592, 0.195978244370824];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p245=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p245.svg"}], p245];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.648515,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.001000,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["246: 0 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.165710258690592, 0.195978244370824];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p246=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p246.svg"}], p246];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.242369,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.407146,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["247: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.165710258690592, 0.195978244370824];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p247=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p247.svg"}], p247];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "3.648515,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.813291,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["248: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.212562118390605, 0.242830104070837];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p248=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p248.svg"}], p248];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.054660,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.219437,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["249: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.618707750657604, 0.648975736337835];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p249=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p249.svg"}], p249];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.460806,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.625583,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["250: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.024853561738536, 1.055121547418768];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p250=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p250.svg"}], p250];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "4.866952,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.031729,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["251: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.430999134400889, 1.461267120081121];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p251=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p251.svg"}], p251];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.273097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.437874,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["252: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.837144945481822, 1.867412931162054];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p252=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p252.svg"}], p252];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "5.679243,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[2.844020,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["253: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.243290518144176, 2.273558503824408];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p253=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p253.svg"}], p253];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.085389,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[3.250166,0,3.241369,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["254: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.649436329225109, 2.679704314905341];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p254=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart-pruned-shifted", "HEARTReduced_p254.svg"}], p254];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["HEARTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["HEARTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["HEARTReduced_convergence_times.csv", ","];
  ];
  WriteString["HEARTReduced_convergence_times.csv", "6.491534,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
