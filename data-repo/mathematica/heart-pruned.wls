#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

HEART[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x0Sneg, x0SnegInitial],
conc[x1Spos, x1SposInitial],
conc[x1Sneg, x1SnegInitial],
rxn[x0Spos,iL0N0Spos+iL0N2Spos+iL0N3Spos+iL0N4Sneg,1],
rxn[x0Sneg,iL0N0Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Spos,1],
rxn[x1Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Sneg+iL0N4Sneg,1],
rxn[x1Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Spos+iL0N4Spos,1],
conc[iL0N0Sneg,2.2061376334],
conc[iL0N1Sneg,0.5232925276],
conc[iL0N2Sneg,0.6654773492],
conc[iL0N3Sneg,2.2042284948],
conc[iL0N4Sneg,2.2514632358],
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

analyzeConvergenceTime = False;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=HEART[1.529019,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p0.svg"}], p0];

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
  WriteString["HEART_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,1.620184];
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
correctResults=List[-0.144568049845011, 0.174836035525243];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p1.svg"}], p1];

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
  WriteString["HEART_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p2.svg"}], p2];

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
  WriteString["HEART_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p3.svg"}], p3];

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
  WriteString["HEART_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p4.svg"}], p4];

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
  WriteString["HEART_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p5.svg"}], p5];

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
  WriteString["HEART_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p6.svg"}], p6];

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
  WriteString["HEART_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p7.svg"}], p7];

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
  WriteString["HEART_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p8.svg"}], p8];

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
  WriteString["HEART_convergence_times.csv", "3.340331,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p9.svg"}], p9];

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
  WriteString["HEART_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p10.svg"}], p10];

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
  WriteString["HEART_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p11.svg"}], p11];

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
  WriteString["HEART_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p12.svg"}], p12];

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
  WriteString["HEART_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p13.svg"}], p13];

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
  WriteString["HEART_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p14.svg"}], p14];

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
  WriteString["HEART_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,1.620184];
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
correctResults=List[-0.144568049845011, 0.174836035525243];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p15.svg"}], p15];

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
  WriteString["HEART_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p16.svg"}], p16];

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
  WriteString["HEART_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p17.svg"}], p17];

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
  WriteString["HEART_convergence_times.csv", "2.917749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,1.388730];
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
correctResults=List[ 0.318341648641317,-0.288073662961085];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p18.svg"}], p18];

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
  WriteString["HEART_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p19.svg"}], p19];

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
  WriteString["HEART_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p20.svg"}], p20];

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
  WriteString["HEART_convergence_times.csv", "1.699312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p21.svg"}], p21];

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
  WriteString["HEART_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p22.svg"}], p22];

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
  WriteString["HEART_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p23.svg"}], p23];

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
  WriteString["HEART_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p24.svg"}], p24];

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
  WriteString["HEART_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p25.svg"}], p25];

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
  WriteString["HEART_convergence_times.csv", "3.108876,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p26.svg"}], p26];

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
  WriteString["HEART_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p27.svg"}], p27];

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
  WriteString["HEART_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p28.svg"}], p28];

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
  WriteString["HEART_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p29.svg"}], p29];

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
  WriteString["HEART_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p30.svg"}], p30];

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
  WriteString["HEART_convergence_times.csv", "1.699312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p31.svg"}], p31];

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
  WriteString["HEART_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,1.388730];
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
correctResults=List[ 0.318341648641317,-0.288073662961085];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p32.svg"}], p32];

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
  WriteString["HEART_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p33.svg"}], p33];

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
  WriteString["HEART_convergence_times.csv", "2.917749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p34.svg"}], p34];

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
  WriteString["HEART_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,1.157275];
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
correctResults=List[ 0.781251585546225,-0.750983599865993];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p35.svg"}], p35];

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
  WriteString["HEART_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p36.svg"}], p36];

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
  WriteString["HEART_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p37.svg"}], p37];

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
  WriteString["HEART_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p38.svg"}], p38];

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
  WriteString["HEART_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p39.svg"}], p39];

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
  WriteString["HEART_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p40.svg"}], p40];

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
  WriteString["HEART_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p41.svg"}], p41];

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
  WriteString["HEART_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p42.svg"}], p42];

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
  WriteString["HEART_convergence_times.csv", "2.877421,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p43.svg"}], p43];

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
  WriteString["HEART_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p44.svg"}], p44];

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
  WriteString["HEART_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p45.svg"}], p45];

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
  WriteString["HEART_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p46.svg"}], p46];

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
  WriteString["HEART_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p47.svg"}], p47];

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
  WriteString["HEART_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p48.svg"}], p48];

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
  WriteString["HEART_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,1.157275];
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
correctResults=List[ 0.781251585546225,-0.750983599865993];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p49.svg"}], p49];

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
  WriteString["HEART_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p50.svg"}], p50];

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
  WriteString["HEART_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p51.svg"}], p51];

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
  WriteString["HEART_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p52.svg"}], p52];

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
  WriteString["HEART_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p53.svg"}], p53];

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
  WriteString["HEART_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p54.svg"}], p54];

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
  WriteString["HEART_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p55.svg"}], p55];

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
  WriteString["HEART_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p56.svg"}], p56];

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
  WriteString["HEART_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p57.svg"}], p57];

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
  WriteString["HEART_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p58.svg"}], p58];

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
  WriteString["HEART_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p59.svg"}], p59];

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
  WriteString["HEART_convergence_times.csv", "2.645966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p60.svg"}], p60];

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
  WriteString["HEART_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p61.svg"}], p61];

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
  WriteString["HEART_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p62.svg"}], p62];

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
  WriteString["HEART_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p63.svg"}], p63];

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
  WriteString["HEART_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p64.svg"}], p64];

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
  WriteString["HEART_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p65.svg"}], p65];

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
  WriteString["HEART_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p66.svg"}], p66];

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
  WriteString["HEART_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p67.svg"}], p67];

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
  WriteString["HEART_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p68.svg"}], p68];

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
  WriteString["HEART_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p69.svg"}], p69];

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
  WriteString["HEART_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p70.svg"}], p70];

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
  WriteString["HEART_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p71.svg"}], p71];

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
  WriteString["HEART_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p72.svg"}], p72];

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
  WriteString["HEART_convergence_times.csv", "0.789928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p73.svg"}], p73];

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
  WriteString["HEART_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p74.svg"}], p74];

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
  WriteString["HEART_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p75.svg"}], p75];

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
  WriteString["HEART_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p76.svg"}], p76];

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
  WriteString["HEART_convergence_times.csv", "2.414511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p77.svg"}], p77];

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
  WriteString["HEART_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p78.svg"}], p78];

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
  WriteString["HEART_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p79.svg"}], p79];

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
  WriteString["HEART_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p80.svg"}], p80];

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
  WriteString["HEART_convergence_times.csv", "0.789928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p81.svg"}], p81];

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
  WriteString["HEART_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p82.svg"}], p82];

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
  WriteString["HEART_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p83.svg"}], p83];

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
  WriteString["HEART_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p84.svg"}], p84];

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
  WriteString["HEART_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p85.svg"}], p85];

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
  WriteString["HEART_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p86.svg"}], p86];

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
  WriteString["HEART_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p87.svg"}], p87];

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
  WriteString["HEART_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p88.svg"}], p88];

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
  WriteString["HEART_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p89.svg"}], p89];

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
  WriteString["HEART_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p90.svg"}], p90];

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
  WriteString["HEART_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p91.svg"}], p91];

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
  WriteString["HEART_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p92.svg"}], p92];

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
  WriteString["HEART_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p93.svg"}], p93];

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
  WriteString["HEART_convergence_times.csv", "2.183056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p94.svg"}], p94];

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
  WriteString["HEART_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p95.svg"}], p95];

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
  WriteString["HEART_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p96.svg"}], p96];

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
  WriteString["HEART_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p97.svg"}], p97];

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
  WriteString["HEART_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p98.svg"}], p98];

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
  WriteString["HEART_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p99.svg"}], p99];

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
  WriteString["HEART_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p100.svg"}], p100];

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
  WriteString["HEART_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p101.svg"}], p101];

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
  WriteString["HEART_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.231455];
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
correctResults=List[ 0.299094656669775,-0.268826670989543];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p102.svg"}], p102];

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
  WriteString["HEART_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.231455];
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
correctResults=List[ 0.705240348541418,-0.674972362861186];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p103.svg"}], p103];

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
  WriteString["HEART_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p104.svg"}], p104];

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
  WriteString["HEART_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p105.svg"}], p105];

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
  WriteString["HEART_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p106.svg"}], p106];

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
  WriteString["HEART_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p107.svg"}], p107];

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
  WriteString["HEART_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p108.svg"}], p108];

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
  WriteString["HEART_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p109.svg"}], p109];

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
  WriteString["HEART_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p110.svg"}], p110];

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
  WriteString["HEART_convergence_times.csv", "1.951602,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p111.svg"}], p111];

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
  WriteString["HEART_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p112.svg"}], p112];

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
  WriteString["HEART_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p113.svg"}], p113];

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
  WriteString["HEART_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p114.svg"}], p114];

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
  WriteString["HEART_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p115.svg"}], p115];

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
  WriteString["HEART_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p116.svg"}], p116];

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
  WriteString["HEART_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,0.231455];
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
correctResults=List[ 0.705240348541418,-0.674972362861186];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p117.svg"}], p117];

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
  WriteString["HEART_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,0.231455];
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
correctResults=List[ 0.299094656669775,-0.268826670989543];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p118.svg"}], p118];

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
  WriteString["HEART_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p119.svg"}], p119];

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
  WriteString["HEART_convergence_times.csv", "1.529019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,-0.000000];
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
correctResults=List[ 0.473785439693609,-0.443517454013377];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p120.svg"}], p120];

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
  WriteString["HEART_convergence_times.csv", "1.122873,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p121.svg"}], p121];

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
  WriteString["HEART_convergence_times.csv", "0.716728,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p122.svg"}], p122];

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
  WriteString["HEART_convergence_times.csv", "0.310582,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p123.svg"}], p123];

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
  WriteString["HEART_convergence_times.csv", "0.095564,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p124.svg"}], p124];

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
  WriteString["HEART_convergence_times.csv", "0.501709,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p125.svg"}], p125];

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
  WriteString["HEART_convergence_times.csv", "0.907855,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p126.svg"}], p126];

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
  WriteString["HEART_convergence_times.csv", "1.314001,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p127.svg"}], p127];

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
  WriteString["HEART_convergence_times.csv", "1.720147,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p128.svg"}], p128];

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
  WriteString["HEART_convergence_times.csv", "1.314001,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p129.svg"}], p129];

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
  WriteString["HEART_convergence_times.csv", "0.907855,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p130.svg"}], p130];

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
  WriteString["HEART_convergence_times.csv", "0.501709,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p131.svg"}], p131];

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
  WriteString["HEART_convergence_times.csv", "0.095564,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p132.svg"}], p132];

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
  WriteString["HEART_convergence_times.csv", "0.310582,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p133.svg"}], p133];

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
  WriteString["HEART_convergence_times.csv", "0.716728,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0,-0.000000];
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
correctResults=List[ 0.473785439693609,-0.443517454013377];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p134.svg"}], p134];

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
  WriteString["HEART_convergence_times.csv", "1.122873,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p135.svg"}], p135];

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
  WriteString["HEART_convergence_times.csv", "1.529019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.231455,0];
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
correctResults=List[-0.163815265333971, 0.194083251014203];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p136.svg"}], p136];

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
  WriteString["HEART_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.231455,0];
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
correctResults=List[ 0.242330426537672,-0.21206244085744 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p137.svg"}], p137];

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
  WriteString["HEART_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.231455,0];
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
correctResults=List[ 0.648476118409315,-0.618208132729083];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p138.svg"}], p138];

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
  WriteString["HEART_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p139.svg"}], p139];

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
  WriteString["HEART_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p140.svg"}], p140];

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
  WriteString["HEART_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p141.svg"}], p141];

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
  WriteString["HEART_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p142.svg"}], p142];

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
  WriteString["HEART_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p143.svg"}], p143];

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
  WriteString["HEART_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p144.svg"}], p144];

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
  WriteString["HEART_convergence_times.csv", "1.951602,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p145.svg"}], p145];

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
  WriteString["HEART_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p146.svg"}], p146];

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
  WriteString["HEART_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p147.svg"}], p147];

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
  WriteString["HEART_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p148.svg"}], p148];

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
  WriteString["HEART_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p149.svg"}], p149];

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
  WriteString["HEART_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.231455,0];
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
correctResults=List[ 0.648476118409315,-0.618208132729083];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p150=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p150.svg"}], p150];

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
  WriteString["HEART_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.231455,0];
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
correctResults=List[ 0.242330426537672,-0.21206244085744 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p151=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p151.svg"}], p151];

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
  WriteString["HEART_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.231455,0];
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
correctResults=List[-0.163815265333971, 0.194083251014203];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p152=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p152.svg"}], p152];

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
  WriteString["HEART_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p153.svg"}], p153];

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
  WriteString["HEART_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.462910,0];
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
correctResults=List[0.010875621997992,0.01939236368224 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p154=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p154.svg"}], p154];

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
  WriteString["HEART_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p155.svg"}], p155];

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
  WriteString["HEART_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.462910,0];
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
correctResults=List[ 0.8231670355436  ,-0.792899049863368];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p156=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p156.svg"}], p156];

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
  WriteString["HEART_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p157.svg"}], p157];

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
  WriteString["HEART_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p158.svg"}], p158];

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
  WriteString["HEART_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p159.svg"}], p159];

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
  WriteString["HEART_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p160.svg"}], p160];

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
  WriteString["HEART_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p161.svg"}], p161];

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
  WriteString["HEART_convergence_times.csv", "2.183056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p162.svg"}], p162];

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
  WriteString["HEART_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p163.svg"}], p163];

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
  WriteString["HEART_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p164.svg"}], p164];

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
  WriteString["HEART_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p165.svg"}], p165];

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
  WriteString["HEART_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.462910,0];
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
correctResults=List[ 0.8231670355436  ,-0.792899049863368];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p166=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p166.svg"}], p166];

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
  WriteString["HEART_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p167.svg"}], p167];

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
  WriteString["HEART_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.462910,0];
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
correctResults=List[0.010875621997992,0.01939236368224 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p168=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p168.svg"}], p168];

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
  WriteString["HEART_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p169.svg"}], p169];

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
  WriteString["HEART_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.694365,0];
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
correctResults=List[-0.797797416070507, 0.828065401750739];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p170=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p170.svg"}], p170];

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
  WriteString["HEART_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.694365,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p171.svg"}], p171];

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
  WriteString["HEART_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.694365,0];
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
correctResults=List[0.01449396767278 ,0.015774018007452];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p172=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p172.svg"}], p172];

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
  WriteString["HEART_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.694365,0];
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
correctResults=List[ 0.420639689346745,-0.390371703666513];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p173=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p173.svg"}], p173];

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
  WriteString["HEART_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p174=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p174.svg"}], p174];

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
  WriteString["HEART_convergence_times.csv", "0.789929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p175=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p175.svg"}], p175];

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
  WriteString["HEART_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p176=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p176.svg"}], p176];

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
  WriteString["HEART_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p177=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p177.svg"}], p177];

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
  WriteString["HEART_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p178=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p178.svg"}], p178];

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
  WriteString["HEART_convergence_times.csv", "2.414511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p179=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p179.svg"}], p179];

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
  WriteString["HEART_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p180=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p180.svg"}], p180];

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
  WriteString["HEART_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p181=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p181.svg"}], p181];

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
  WriteString["HEART_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p182=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p182.svg"}], p182];

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
  WriteString["HEART_convergence_times.csv", "0.789929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.694365,0];
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
correctResults=List[ 0.420639689346745,-0.390371703666513];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p183=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p183.svg"}], p183];

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
  WriteString["HEART_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.694365,0];
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
correctResults=List[0.01449396767278 ,0.015774018007452];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p184=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p184.svg"}], p184];

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
  WriteString["HEART_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.694365,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p185.svg"}], p185];

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
  WriteString["HEART_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.694365,0];
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
correctResults=List[-0.797797416070507, 0.828065401750739];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p186=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p186.svg"}], p186];

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
  WriteString["HEART_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.925820,0];
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
correctResults=List[-1.260706995347545, 1.290974981027777];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p187=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p187.svg"}], p187];

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
  WriteString["HEART_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.925820,0];
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
correctResults=List[-0.854561303475902, 0.884829289156134];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p188=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p188.svg"}], p188];

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
  WriteString["HEART_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.925820,0];
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
correctResults=List[-0.448415611604259, 0.478683597284491];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p189=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p189.svg"}], p189];

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
  WriteString["HEART_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.925820,0];
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
correctResults=List[-0.042269889930293, 0.072537875610525];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p190=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p190.svg"}], p190];

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
  WriteString["HEART_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.925820,0];
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
correctResults=List[ 0.363875816842511,-0.333607831162279];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p191=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p191.svg"}], p191];

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
  WriteString["HEART_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p192=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p192.svg"}], p192];

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
  WriteString["HEART_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p193=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p193.svg"}], p193];

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
  WriteString["HEART_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p194=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p194.svg"}], p194];

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
  WriteString["HEART_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p195=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p195.svg"}], p195];

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
  WriteString["HEART_convergence_times.csv", "2.645966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p196=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p196.svg"}], p196];

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
  WriteString["HEART_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p197=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p197.svg"}], p197];

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
  WriteString["HEART_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p198=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p198.svg"}], p198];

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
  WriteString["HEART_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,0.925820,0];
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
correctResults=List[ 0.363875816842511,-0.333607831162279];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p199=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p199.svg"}], p199];

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
  WriteString["HEART_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,0.925820,0];
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
correctResults=List[-0.042269889930293, 0.072537875610525];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p200=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p200.svg"}], p200];

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
  WriteString["HEART_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,0.925820,0];
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
correctResults=List[-0.448415611604259, 0.478683597284491];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p201=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p201.svg"}], p201];

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
  WriteString["HEART_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,0.925820,0];
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
correctResults=List[-0.854561303475902, 0.884829289156134];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p202=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p202.svg"}], p202];

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
  WriteString["HEART_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,0.925820,0];
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
correctResults=List[-1.260706995347545, 1.290974981027777];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p203=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p203.svg"}], p203];

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
  WriteString["HEART_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p204.svg"}], p204];

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
  WriteString["HEART_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,1.157275,0];
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
correctResults=List[-1.317471240380809, 1.347739226061041];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p205=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p205.svg"}], p205];

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
  WriteString["HEART_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p206.svg"}], p206];

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
  WriteString["HEART_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,1.157275,0];
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
correctResults=List[-0.505179826835201, 0.535447812515433];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p207=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p207.svg"}], p207];

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
  WriteString["HEART_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,1.157275,0];
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
correctResults=List[-0.099034120062397, 0.129302105742628];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p208=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p208.svg"}], p208];

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
  WriteString["HEART_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p209.svg"}], p209];

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
  WriteString["HEART_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p210.svg"}], p210];

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
  WriteString["HEART_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p211.svg"}], p211];

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
  WriteString["HEART_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p212.svg"}], p212];

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
  WriteString["HEART_convergence_times.csv", "2.877421,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p213.svg"}], p213];

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
  WriteString["HEART_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p214.svg"}], p214];

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
  WriteString["HEART_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p215.svg"}], p215];

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
  WriteString["HEART_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,1.157275,0];
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
correctResults=List[-0.099034120062397, 0.129302105742628];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p216=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p216.svg"}], p216];

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
  WriteString["HEART_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,1.157275,0];
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
correctResults=List[-0.505179826835201, 0.535447812515433];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p217=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p217.svg"}], p217];

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
  WriteString["HEART_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p218.svg"}], p218];

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
  WriteString["HEART_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,1.157275,0];
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
correctResults=List[-1.317471240380809, 1.347739226061041];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p219=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p219.svg"}], p219];

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
  WriteString["HEART_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p220.svg"}], p220];

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
  WriteString["HEART_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p221.svg"}], p221];

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
  WriteString["HEART_convergence_times.csv", "2.917748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,1.388729,0];
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
correctResults=List[-1.780380938867137, 1.810648924547369];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p222=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p222.svg"}], p222];

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
  WriteString["HEART_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p223.svg"}], p223];

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
  WriteString["HEART_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,1.388729,0];
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
correctResults=List[-0.968089525321529, 0.998357511001761];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p224=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p224.svg"}], p224];

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
  WriteString["HEART_convergence_times.csv", "1.699311,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,1.388729,0];
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
correctResults=List[-0.561943818548725, 0.592211804228957];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p225=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p225.svg"}], p225];

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
  WriteString["HEART_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,1.388729,0];
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
correctResults=List[-0.15579811177592 , 0.186066097456152];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p226=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p226.svg"}], p226];

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
  WriteString["HEART_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p227.svg"}], p227];

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
  WriteString["HEART_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p228.svg"}], p228];

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
  WriteString["HEART_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p229.svg"}], p229];

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
  WriteString["HEART_convergence_times.csv", "3.108876,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p230.svg"}], p230];

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
  WriteString["HEART_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p231.svg"}], p231];

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
  WriteString["HEART_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,1.388729,0];
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
correctResults=List[-0.15579811177592 , 0.186066097456152];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p232=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p232.svg"}], p232];

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
  WriteString["HEART_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,1.388729,0];
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
correctResults=List[-0.561943818548725, 0.592211804228957];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p233=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p233.svg"}], p233];

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
  WriteString["HEART_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,1.388729,0];
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
correctResults=List[-0.968089525321529, 0.998357511001761];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p234=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p234.svg"}], p234];

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
  WriteString["HEART_convergence_times.csv", "1.699311,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p235.svg"}], p235];

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
  WriteString["HEART_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,1.388729,0];
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
correctResults=List[-1.780380938867137, 1.810648924547369];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p236=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p236.svg"}], p236];

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
  WriteString["HEART_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p237.svg"}], p237];

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
  WriteString["HEART_convergence_times.csv", "2.917748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,1.620184,0];
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
correctResults=List[-2.649436567643688, 2.67970455332392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p238=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p238.svg"}], p238];

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
  WriteString["HEART_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,1.620184,0];
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
correctResults=List[-2.243290875772045, 2.273558861452277];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p239=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p239.svg"}], p239];

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
  WriteString["HEART_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,1.620184,0];
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
correctResults=List[-1.837145183900401, 1.867413169580633];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p240=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p240.svg"}], p240];

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
  WriteString["HEART_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,1.620184,0];
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
correctResults=List[-1.430999462226436, 1.461267447906668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p241=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p241.svg"}], p241];

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
  WriteString["HEART_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,1.620184,0];
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
correctResults=List[-1.024853755453632, 1.055121741133864];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p242=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p242.svg"}], p242];

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
  WriteString["HEART_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,1.620184,0];
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
correctResults=List[-0.618708048680827, 0.648976034361059];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p243=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p243.svg"}], p243];

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
  WriteString["HEART_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,1.620184,0];
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
correctResults=List[-0.212562356809184, 0.242830342489416];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p244=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p244.svg"}], p244];

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
  WriteString["HEART_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,1.620184,0];
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
correctResults=List[-0.165710377899881, 0.195978363580113];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p245=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p245.svg"}], p245];

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
  WriteString["HEART_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.720147,1.620184,0];
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
correctResults=List[-0.165710377899881, 0.195978363580113];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p246=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p246.svg"}], p246];

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
  WriteString["HEART_convergence_times.csv", "3.340331,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,1.314001,1.620184,0];
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
correctResults=List[-0.165710377899881, 0.195978363580113];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p247=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p247.svg"}], p247];

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
  WriteString["HEART_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.907855,1.620184,0];
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
correctResults=List[-0.212562356809184, 0.242830342489416];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p248=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p248.svg"}], p248];

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
  WriteString["HEART_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.501709,1.620184,0];
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
correctResults=List[-0.618708048680827, 0.648976034361059];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p249=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p249.svg"}], p249];

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
  WriteString["HEART_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0,0.095564,1.620184,0];
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
correctResults=List[-1.024853755453632, 1.055121741133864];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p250=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p250.svg"}], p250];

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
  WriteString["HEART_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.310582,0,1.620184,0];
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
correctResults=List[-1.430999462226436, 1.461267447906668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p251=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p251.svg"}], p251];

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
  WriteString["HEART_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[0.716728,0,1.620184,0];
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
correctResults=List[-1.837145183900401, 1.867413169580633];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p252=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p252.svg"}], p252];

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
  WriteString["HEART_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.122873,0,1.620184,0];
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
correctResults=List[-2.243290875772045, 2.273558861452277];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p253=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p253.svg"}], p253];

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
  WriteString["HEART_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEART_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEART_convergence_times.csv", ""];
];

rsys=HEART[1.529019,0,1.620184,0];
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
correctResults=List[-2.649436567643688, 2.67970455332392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p254=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEART_p254.svg"}], p254];

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
  WriteString["HEART_convergence_times.csv", "3.149204,"];
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

analyzeConvergenceTime = False;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];
rsys=HEARTReduced[1.529019,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p0.svg"}], p0];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,1.620184];
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
correctResults=List[-0.144568049845011, 0.174836035525243];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p1.svg"}], p1];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p2.svg"}], p2];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p3.svg"}], p3];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p4.svg"}], p4];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p5.svg"}], p5];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p6.svg"}], p6];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p7.svg"}], p7];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p8.svg"}], p8];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.340331,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p9.svg"}], p9];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p10.svg"}], p10];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p11.svg"}], p11];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p12.svg"}], p12];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p13.svg"}], p13];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p14.svg"}], p14];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,1.620184];
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
correctResults=List[-0.144568049845011, 0.174836035525243];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p15.svg"}], p15];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,1.620184];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p16.svg"}], p16];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p17.svg"}], p17];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.917749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,1.388730];
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
correctResults=List[ 0.318341648641317,-0.288073662961085];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p18.svg"}], p18];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p19.svg"}], p19];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p20.svg"}], p20];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.699312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p21.svg"}], p21];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p22.svg"}], p22];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p23.svg"}], p23];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p24.svg"}], p24];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p25.svg"}], p25];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.108876,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p26.svg"}], p26];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p27.svg"}], p27];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p28.svg"}], p28];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p29.svg"}], p29];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p30.svg"}], p30];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.699312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p31.svg"}], p31];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,1.388730];
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
correctResults=List[ 0.318341648641317,-0.288073662961085];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p32.svg"}], p32];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,1.388730];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p33.svg"}], p33];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.917749,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p34.svg"}], p34];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,1.157275];
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
correctResults=List[ 0.781251585546225,-0.750983599865993];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p35.svg"}], p35];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p36.svg"}], p36];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p37.svg"}], p37];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p38.svg"}], p38];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p39.svg"}], p39];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p40.svg"}], p40];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p41.svg"}], p41];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p42.svg"}], p42];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.877421,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p43.svg"}], p43];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p44.svg"}], p44];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p45.svg"}], p45];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p46.svg"}], p46];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p47.svg"}], p47];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p48.svg"}], p48];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,1.157275];
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
correctResults=List[ 0.781251585546225,-0.750983599865993];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p49.svg"}], p49];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,1.157275];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p50.svg"}], p50];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p51.svg"}], p51];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p52.svg"}], p52];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p53.svg"}], p53];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p54.svg"}], p54];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p55.svg"}], p55];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p56.svg"}], p56];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p57.svg"}], p57];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p58.svg"}], p58];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p59.svg"}], p59];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.645966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p60.svg"}], p60];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p61.svg"}], p61];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p62.svg"}], p62];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p63.svg"}], p63];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p64.svg"}], p64];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p65.svg"}], p65];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p66.svg"}], p66];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.925820];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p67.svg"}], p67];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p68.svg"}], p68];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p69.svg"}], p69];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p70.svg"}], p70];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p71.svg"}], p71];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p72.svg"}], p72];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.789928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p73.svg"}], p73];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p74.svg"}], p74];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p75.svg"}], p75];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p76.svg"}], p76];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.414511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p77.svg"}], p77];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p78.svg"}], p78];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p79.svg"}], p79];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p80.svg"}], p80];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.789928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p81.svg"}], p81];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p82.svg"}], p82];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p83.svg"}], p83];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.694365];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p84.svg"}], p84];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p85.svg"}], p85];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p86.svg"}], p86];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p87.svg"}], p87];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p88.svg"}], p88];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p89.svg"}], p89];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p90.svg"}], p90];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p91.svg"}], p91];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p92.svg"}], p92];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p93.svg"}], p93];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.183056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p94.svg"}], p94];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p95.svg"}], p95];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p96.svg"}], p96];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p97.svg"}], p97];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p98.svg"}], p98];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p99.svg"}], p99];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p100.svg"}], p100];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.462910];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p101.svg"}], p101];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.231455];
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
correctResults=List[ 0.299094656669775,-0.268826670989543];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p102=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p102.svg"}], p102];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.231455];
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
correctResults=List[ 0.705240348541418,-0.674972362861186];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p103=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p103.svg"}], p103];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p104.svg"}], p104];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p105.svg"}], p105];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p106.svg"}], p106];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p107.svg"}], p107];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p108.svg"}], p108];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p109.svg"}], p109];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p110.svg"}], p110];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.951602,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p111.svg"}], p111];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p112.svg"}], p112];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p113.svg"}], p113];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p114.svg"}], p114];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p115.svg"}], p115];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,0.231455];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p116.svg"}], p116];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,0.231455];
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
correctResults=List[ 0.705240348541418,-0.674972362861186];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p117=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p117.svg"}], p117];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,0.231455];
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
correctResults=List[ 0.299094656669775,-0.268826670989543];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p118=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p118.svg"}], p118];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p119.svg"}], p119];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.529019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,-0.000000];
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
correctResults=List[ 0.473785439693609,-0.443517454013377];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p120=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p120.svg"}], p120];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.122873,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p121.svg"}], p121];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.716728,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p122.svg"}], p122];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.310582,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p123.svg"}], p123];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.095564,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p124.svg"}], p124];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.501709,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p125.svg"}], p125];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.907855,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p126.svg"}], p126];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.314001,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p127.svg"}], p127];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.720147,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p128.svg"}], p128];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.314001,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p129.svg"}], p129];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.907855,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p130.svg"}], p130];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.501709,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p131.svg"}], p131];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.095564,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p132.svg"}], p132];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.310582,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p133.svg"}], p133];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.716728,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0,-0.000000];
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
correctResults=List[ 0.473785439693609,-0.443517454013377];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p134=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p134.svg"}], p134];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.122873,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0,-0.000000];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p135.svg"}], p135];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.529019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.231455,0];
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
correctResults=List[-0.163815265333971, 0.194083251014203];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p136=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p136.svg"}], p136];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.231455,0];
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
correctResults=List[ 0.242330426537672,-0.21206244085744 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p137=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p137.svg"}], p137];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.231455,0];
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
correctResults=List[ 0.648476118409315,-0.618208132729083];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p138=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p138.svg"}], p138];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p139.svg"}], p139];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p140.svg"}], p140];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p141.svg"}], p141];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p142.svg"}], p142];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p143.svg"}], p143];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p144.svg"}], p144];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.951602,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p145.svg"}], p145];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.545456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p146.svg"}], p146];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.139310,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p147.svg"}], p147];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.733164,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p148.svg"}], p148];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.327019,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.231455,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p149.svg"}], p149];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.542037,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.231455,0];
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
correctResults=List[ 0.648476118409315,-0.618208132729083];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p150=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p150.svg"}], p150];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.948183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.231455,0];
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
correctResults=List[ 0.242330426537672,-0.21206244085744 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p151=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p151.svg"}], p151];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.354328,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.231455,0];
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
correctResults=List[-0.163815265333971, 0.194083251014203];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p152=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p152.svg"}], p152];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.760474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p153.svg"}], p153];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.462910,0];
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
correctResults=List[0.010875621997992,0.01939236368224 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p154=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p154.svg"}], p154];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p155.svg"}], p155];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.462910,0];
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
correctResults=List[ 0.8231670355436  ,-0.792899049863368];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p156=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p156.svg"}], p156];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p157.svg"}], p157];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p158.svg"}], p158];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p159.svg"}], p159];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p160.svg"}], p160];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p161.svg"}], p161];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.183056,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p162.svg"}], p162];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.776911,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p163.svg"}], p163];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.370765,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p164.svg"}], p164];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.964619,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p165.svg"}], p165];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.558474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.462910,0];
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
correctResults=List[ 0.8231670355436  ,-0.792899049863368];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p166=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p166.svg"}], p166];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.773492,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p167.svg"}], p167];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.179638,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.462910,0];
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
correctResults=List[0.010875621997992,0.01939236368224 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p168=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p168.svg"}], p168];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.585783,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.462910,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p169.svg"}], p169];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.991929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.694365,0];
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
correctResults=List[-0.797797416070507, 0.828065401750739];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p170=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p170.svg"}], p170];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.694365,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p171.svg"}], p171];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.694365,0];
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
correctResults=List[0.01449396767278 ,0.015774018007452];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p172=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p172.svg"}], p172];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.694365,0];
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
correctResults=List[ 0.420639689346745,-0.390371703666513];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p173=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p173.svg"}], p173];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p174=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p174.svg"}], p174];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.789929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p175=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p175.svg"}], p175];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p176=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p176.svg"}], p176];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p177=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p177.svg"}], p177];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p178=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p178.svg"}], p178];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.414511,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p179=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p179.svg"}], p179];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.008366,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p180=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p180.svg"}], p180];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.602220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p181=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p181.svg"}], p181];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.196074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.694365,0];
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
correctResults=List[ 0.760109197886709,-0.729841212206477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p182=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p182.svg"}], p182];

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
  WriteString["HEARTReduced_convergence_times.csv", "0.789929,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.694365,0];
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
correctResults=List[ 0.420639689346745,-0.390371703666513];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p183=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p183.svg"}], p183];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.004947,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.694365,0];
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
correctResults=List[0.01449396767278 ,0.015774018007452];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p184=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p184.svg"}], p184];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.411093,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.694365,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p185.svg"}], p185];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.817238,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.694365,0];
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
correctResults=List[-0.797797416070507, 0.828065401750739];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p186=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p186.svg"}], p186];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.223384,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.925820,0];
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
correctResults=List[-1.260706995347545, 1.290974981027777];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p187=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p187.svg"}], p187];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.925820,0];
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
correctResults=List[-0.854561303475902, 0.884829289156134];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p188=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p188.svg"}], p188];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.925820,0];
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
correctResults=List[-0.448415611604259, 0.478683597284491];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p189=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p189.svg"}], p189];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.925820,0];
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
correctResults=List[-0.042269889930293, 0.072537875610525];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p190=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p190.svg"}], p190];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.925820,0];
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
correctResults=List[ 0.363875816842511,-0.333607831162279];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p191=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p191.svg"}], p191];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p192=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p192.svg"}], p192];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p193=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p193.svg"}], p193];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p194=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p194.svg"}], p194];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p195=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p195.svg"}], p195];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.645966,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p196=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p196.svg"}], p196];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.239820,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p197=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p197.svg"}], p197];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.833675,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,0.925820,0];
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
correctResults=List[ 0.52865440824819 ,-0.498386422567958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p198=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p198.svg"}], p198];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.427529,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,0.925820,0];
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
correctResults=List[ 0.363875816842511,-0.333607831162279];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p199=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p199.svg"}], p199];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.021383,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,0.925820,0];
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
correctResults=List[-0.042269889930293, 0.072537875610525];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p200=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p200.svg"}], p200];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.236402,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,0.925820,0];
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
correctResults=List[-0.448415611604259, 0.478683597284491];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p201=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p201.svg"}], p201];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.642547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,0.925820,0];
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
correctResults=List[-0.854561303475902, 0.884829289156134];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p202=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p202.svg"}], p202];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.048693,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,0.925820,0];
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
correctResults=List[-1.260706995347545, 1.290974981027777];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p203=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p203.svg"}], p203];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.454839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p204.svg"}], p204];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,1.157275,0];
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
correctResults=List[-1.317471240380809, 1.347739226061041];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p205=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p205.svg"}], p205];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p206.svg"}], p206];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,1.157275,0];
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
correctResults=List[-0.505179826835201, 0.535447812515433];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p207=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p207.svg"}], p207];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,1.157275,0];
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
correctResults=List[-0.099034120062397, 0.129302105742628];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p208=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p208.svg"}], p208];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p209.svg"}], p209];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p210.svg"}], p210];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p211.svg"}], p211];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p212.svg"}], p212];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.877421,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p213.svg"}], p213];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.471275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p214.svg"}], p214];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.065130,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p215.svg"}], p215];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.658984,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,1.157275,0];
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
correctResults=List[-0.099034120062397, 0.129302105742628];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p216=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p216.svg"}], p216];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.252838,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,1.157275,0];
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
correctResults=List[-0.505179826835201, 0.535447812515433];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p217=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p217.svg"}], p217];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.467857,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p218.svg"}], p218];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.874002,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,1.157275,0];
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
correctResults=List[-1.317471240380809, 1.347739226061041];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p219=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p219.svg"}], p219];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.280148,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,1.157275,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p220.svg"}], p220];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.686294,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p221.svg"}], p221];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.917748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,1.388729,0];
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
correctResults=List[-1.780380938867137, 1.810648924547369];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p222=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p222.svg"}], p222];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p223.svg"}], p223];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,1.388729,0];
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
correctResults=List[-0.968089525321529, 0.998357511001761];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p224=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p224.svg"}], p224];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.699311,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,1.388729,0];
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
correctResults=List[-0.561943818548725, 0.592211804228957];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p225=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p225.svg"}], p225];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,1.388729,0];
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
correctResults=List[-0.15579811177592 , 0.186066097456152];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p226=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p226.svg"}], p226];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p227.svg"}], p227];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p228.svg"}], p228];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p229.svg"}], p229];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.108876,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p230.svg"}], p230];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.702730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p231.svg"}], p231];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.296585,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,1.388729,0];
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
correctResults=List[-0.15579811177592 , 0.186066097456152];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p232=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p232.svg"}], p232];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.890439,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,1.388729,0];
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
correctResults=List[-0.561943818548725, 0.592211804228957];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p233=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p233.svg"}], p233];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.484293,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,1.388729,0];
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
correctResults=List[-0.968089525321529, 0.998357511001761];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p234=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p234.svg"}], p234];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.699311,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p235.svg"}], p235];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.105457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,1.388729,0];
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
correctResults=List[-1.780380938867137, 1.810648924547369];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p236=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p236.svg"}], p236];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.511603,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,1.388729,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p237.svg"}], p237];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.917748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,1.620184,0];
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
correctResults=List[-2.649436567643688, 2.67970455332392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p238=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p238.svg"}], p238];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,1.620184,0];
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
correctResults=List[-2.243290875772045, 2.273558861452277];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p239=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p239.svg"}], p239];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,1.620184,0];
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
correctResults=List[-1.837145183900401, 1.867413169580633];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p240=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p240.svg"}], p240];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,1.620184,0];
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
correctResults=List[-1.430999462226436, 1.461267447906668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p241=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p241.svg"}], p241];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,1.620184,0];
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
correctResults=List[-1.024853755453632, 1.055121741133864];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p242=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p242.svg"}], p242];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,1.620184,0];
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
correctResults=List[-0.618708048680827, 0.648976034361059];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p243=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p243.svg"}], p243];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,1.620184,0];
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
correctResults=List[-0.212562356809184, 0.242830342489416];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p244=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p244.svg"}], p244];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,1.620184,0];
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
correctResults=List[-0.165710377899881, 0.195978363580113];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p245=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p245.svg"}], p245];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.720147,1.620184,0];
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
correctResults=List[-0.165710377899881, 0.195978363580113];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p246=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p246.svg"}], p246];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.340331,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,1.314001,1.620184,0];
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
correctResults=List[-0.165710377899881, 0.195978363580113];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p247=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p247.svg"}], p247];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.934185,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.907855,1.620184,0];
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
correctResults=List[-0.212562356809184, 0.242830342489416];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p248=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p248.svg"}], p248];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.528039,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.501709,1.620184,0];
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
correctResults=List[-0.618708048680827, 0.648976034361059];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p249=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p249.svg"}], p249];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.121894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0,0.095564,1.620184,0];
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
correctResults=List[-1.024853755453632, 1.055121741133864];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p250=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p250.svg"}], p250];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.715748,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.310582,0,1.620184,0];
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
correctResults=List[-1.430999462226436, 1.461267447906668];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p251=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p251.svg"}], p251];

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
  WriteString["HEARTReduced_convergence_times.csv", "1.930766,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[0.716728,0,1.620184,0];
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
correctResults=List[-1.837145183900401, 1.867413169580633];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p252=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p252.svg"}], p252];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.336912,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.122873,0,1.620184,0];
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
correctResults=List[-2.243290875772045, 2.273558861452277];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p253=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p253.svg"}], p253];

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
  WriteString["HEARTReduced_convergence_times.csv", "2.743058,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

rsys=HEARTReduced[1.529019,0,1.620184,0];
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
correctResults=List[-2.649436567643688, 2.67970455332392 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p254=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/heart", "HEARTReduced_p254.svg"}], p254];

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
  WriteString["HEARTReduced_convergence_times.csv", "3.149204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["HEARTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["HEARTReduced_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
