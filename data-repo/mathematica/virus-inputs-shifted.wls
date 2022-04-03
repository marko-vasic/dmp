#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

VIRUS[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x1Spos, x1SposInitial],
conc[x2Spos, x2SposInitial],
conc[x3Spos, x3SposInitial],
conc[x4Spos, x4SposInitial],
conc[x5Spos, x5SposInitial],
conc[x6Spos, x6SposInitial],
conc[x7Spos, x7SposInitial],
conc[x8Spos, x8SposInitial],
conc[x9Spos, x9SposInitial],
rxn[x0Spos,iL0N1Spos+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Spos,1],
rxn[x1Spos,iL0N0Sneg+iL0N1Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Sneg+iL0N6Spos,1],
rxn[x2Spos,iL0N0Sneg+iL0N1Spos+iL0N5Sneg+iL0N7Spos,1],
rxn[x3Spos,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N7Sneg,1],
rxn[x4Spos,iL0N1Spos+iL0N2Spos+iL0N4Sneg+iL0N5Spos,1],
rxn[x5Spos,iL0N0Sneg+iL0N2Spos+iL0N3Sneg+iL0N5Sneg+iL0N7Sneg,1],
rxn[x6Spos,iL0N2Spos+iL0N4Spos+iL0N6Spos+iL0N7Sneg,1],
rxn[x7Spos,iL0N0Spos+iL0N2Spos+iL0N4Spos+iL0N5Spos+iL0N6Spos,1],
rxn[x8Spos,iL0N0Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Spos+iL0N7Sneg,1],
rxn[x9Spos,iL0N1Spos+iL0N2Spos+iL0N4Spos+iL0N6Spos,1],
conc[iL0N0Spos,51.4975256509],
conc[iL0N1Sneg,38.2918576322],
conc[iL0N2Sneg,62.8004886307],
conc[iL0N3Spos,67.8686064535],
conc[iL0N4Sneg,98.4274914505],
conc[iL0N5Sneg,39.7265810318],
conc[iL0N6Sneg,103.0647708898],
conc[iL0N7Spos,57.6742839980],
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
rxn[hL0N4Spos,hL1N1Spos+hL1N2Sneg+hL1N3Sneg,1],
rxn[hL0N4Sneg,hL1N1Sneg+hL1N2Spos+hL1N3Spos,1],
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

VIRUSReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_] =
  reduceFFNCCrn[VIRUS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial], keepInputLayer=True, inputPrefix="x"];

cnt = Count[VIRUS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[VIRUSReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial], rxn[___]];
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
rsys=VIRUS[11.461777,0,9.516586,0,11.103571,0,14.458374,0,19.159218,0,10.870099,0,19.596327,0,16.100349,0,23.158102,0,19.741179,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.178133221687052,-2.456942860491973,-0.042613379244421,
 -1.217998035260949];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "155.165588,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.689230,0,10.593153,0,12.171078,0,15.889871,0,20.779415,0,12.614514,0,22.152349,0,18.841770,0,24.272554,0,21.741167,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-14.931818514899476, 10.43665144429058 ,-15.212276504198924,
  -9.055711594156458];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "170.745102,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.870010,0,11.199254,0,11.230225,0,16.371553,0,20.645689,0,10.977923,0,20.521309,0,18.304695,0,23.573792,0,21.831076,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-7.607293548191322, 7.071883416031302,-8.159970416219576,
 -9.378481625617253];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "166.525528,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.611319,0,8.482581,0,9.677068,0,13.206181,0,17.961912,0,10.433263,0,19.775127,0,15.779161,0,22.363163,0,19.381548,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.587063900376973,-7.107708165587677,-4.03571016661069 ,
 -8.983195157324973];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "146.671326,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.046584,0,8.246875,0,8.809737,0,14.268229,0,19.272160,0,11.980596,0,19.027927,0,15.700846,0,24.358320,0,18.898981,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.487207228230816,-1.7619898790632  , 1.266460627299344,
 -1.137166078947168];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.610260,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.941411,0,8.693128,0,9.600100,0,13.298091,0,18.385958,0,10.230643,0,19.808397,0,15.128860,0,22.273928,0,18.348526,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.752140156175313,-6.856720158995881,-3.541480849899263,
 -8.148271413123314];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "145.709045,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.211110,0,8.406535,0,9.704639,0,13.513978,0,19.091314,0,10.056690,0,18.785633,0,16.185270,0,23.009378,0,19.750353,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.802646747972187,-7.446148106993927,-4.352478812850925,
 -9.198778004920188];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "147.714905,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.519230,0,9.078656,0,9.708947,0,13.598888,0,19.233587,0,10.516981,0,19.658043,0,16.267925,0,21.594252,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.970306507493672,-6.337656209410919,-4.630453895248386,
 -8.366437764441672];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.354385,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.025446,0,9.686650,0,10.435553,0,14.981833,0,19.092472,0,11.411503,0,20.989389,0,16.056190,0,22.409515,0,19.987049,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.761989879063199, 0.652339602184352,
 -0.523045053832177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "157.075592,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.581954,0,8.835573,0,10.389889,0,13.933559,0,18.572113,0,9.487272,0,19.131042,0,16.267925,0,22.451406,0,19.453108,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.434397808458027,-7.401723096312774,-5.076303313888523,
 -8.830529065406028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.103836,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.362940,0,9.840067,0,11.157850,0,14.665680,0,20.248436,0,11.183149,0,20.402449,0,16.605909,0,23.361115,0,20.539339,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.534220851536968,-1.761989879063199, 1.013126286105293,
 -0.883831737753117];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "159.366928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.170544,0,9.283906,0,11.619949,0,14.620515,0,19.554855,0,11.117999,0,18.845320,0,15.601018,0,24.763102,0,20.355854,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.378010110864015,-1.761989879063199, 1.157263509932543,
 -1.027968961580367];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "156.933060,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.372672,0,8.752905,0,9.818369,0,13.924300,0,19.147902,0,10.324461,0,19.361906,0,15.514776,0,22.375805,0,18.247608,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.568019977952657,-6.386634061278595,-2.67030317655942 ,
 -6.964151234900656];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "146.840698,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.332519,0,11.519520,0,11.087201,0,15.247402,0,19.449999,0,10.913099,0,20.976437,0,17.816309,0,23.376234,0,21.649426,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-5.903618690068518, 7.884038185928763,-9.41854830473789 ,
 -9.744213276893861];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "164.368149,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.768849,0,10.242999,0,10.984385,0,14.643098,0,20.410109,0,11.999165,0,20.734142,0,16.321520,0,23.686577,0,19.987049,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.112634727602254,-1.7619898790632  , 0.412791077697922,
 -0.283496529345747];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "160.777878,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.689230,0,10.235244,0,11.809211,0,16.886658,0,21.152651,0,12.206345,0,21.578108,0,18.841770,0,24.492174,0,22.276945,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-15.313951045111878, 10.706696724747124,-12.783066795031443,
 -12.025011864237024];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "171.168335,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.936333,0,9.909870,0,11.491286,0,15.199302,0,19.969896,0,12.336646,0,21.738873,0,17.085239,0,22.417200,0,19.739344,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.80696112835909 ,-1.7619898790632  ,-1.281535323058913,
  1.410829871411089];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "160.823990,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.555123,0,9.667543,0,10.233367,0,14.123928,0,20.109627,0,11.509556,0,20.453497,0,17.288481,0,23.561396,0,21.091629,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.683229952887753,-1.209091663362073, 0.410188367064569,
 -1.386690250114646];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "159.594147,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.399678,0,13.420491,0,12.671948,0,17.706398,0,21.415947,0,13.054283,0,23.535759,0,16.940119,0,22.637810,0,20.133837,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -9.04938876300553 , 11.171586251114311,-14.454616267384374,
 -11.283241444618472];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "173.916275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.337403,0,9.167566,0,10.311484,0,11.733131,0,18.197262,0,11.312148,0,19.748459,0,14.193225,0,22.544113,0,18.091097,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.171382061387716,-8.184150884093537,-2.61788446775815 ,
 -7.567513318335717];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "144.635895,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.728099,0,8.873028,0,9.680514,0,14.781528,0,19.210724,0,9.905540,0,20.034182,0,16.267925,0,21.692162,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.789230457688984,-4.591627309264434,-3.121726821578952,
 -6.185361714636985];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "151.171768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.521747,0,11.594810,0,11.687728,0,15.877000,0,20.001999,0,11.464275,0,20.445372,0,17.628351,0,22.413731,0,21.831076,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-4.694120371864559, 5.490919327591361,-6.493525138566846,
 -7.8829987263901  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "165.466080,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.195474,0,9.308120,0,10.889609,0,15.293469,0,20.644304,0,11.647675,0,20.876877,0,16.488153,0,23.868517,0,20.361359,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.602022239809285,-1.7619898790632  ,-1.076596434509109,
  1.205890982861284];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "159.573547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.140766,0,9.547988,0,11.032059,0,14.070408,0,19.370548,0,12.244457,0,20.667601,0,16.668751,0,23.419861,0,20.031084,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-1.843864509706747,-1.761989879063199,-0.31843870440657 ,
  0.447733252758745];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "158.193527,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.021663,0,8.364161,0,10.300283,0,15.292567,0,19.955807,0,11.737909,0,19.892210,0,16.016184,0,23.160334,0,20.128332,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115824,-1.7619898790632  , 0.652339602184352,
 -0.523045053832176];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "154.869446,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.611319,0,8.254252,0,9.649498,0,13.580144,0,19.279549,0,10.527406,0,19.694870,0,15.160186,0,22.906012,0,19.935673,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.472138515855488,-5.36681480163504 ,-1.70341093412778 ,
 -5.868269772803489];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.598892,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.519230,0,8.854301,0,9.964265,0,13.822001,0,16.405228,0,9.873616,0,19.457401,0,16.715174,0,23.112494,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.772935978318867,-4.821990201415313,-3.306718658126803,
 -6.169067235266868];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.083237,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.683317,0,9.610036,0,10.522286,0,13.898331,0,19.693665,0,11.672759,0,20.836243,0,16.188856,0,23.160830,0,20.908146,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.797807849522808,-1.761989879063199, 0.741997631625312,
 -0.612703083273137];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "158.174469,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.087977,0,8.711477,0,9.708947,0,14.086216,0,18.672581,0,10.189924,0,19.746681,0,15.128860,0,23.090681,0,18.583387,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.93811236992043 ,-3.734049031676544,-0.456528495468112,
 -3.334243626868431];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.006729,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.108665,0,10.071800,0,11.585485,0,16.458496,0,21.244343,0,12.796286,0,21.013008,0,18.765720,0,25.263565,0,20.783375,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-11.251672210300697,  5.972169136856498, -9.114759577901706,
  -6.224263905585514];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "169.090744,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.710974,0,11.688827,0,11.859183,0,15.741957,0,20.254902,0,10.556398,0,21.355621,0,17.741013,0,23.104811,0,21.465940,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -6.564314806984189,  8.367173409317436, -9.028920760820265,
 -11.10011126758883 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "166.479630,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.554701,0,10.992113,0,11.659007,0,17.518061,0,20.645689,0,11.204974,0,19.762682,0,18.060503,0,25.030560,0,21.100805,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -6.018456083275204,  6.620846962784232, -6.021720223947503,
 -10.614658911395185];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.529083,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.479728,0,9.059929,0,10.077707,0,13.732801,0,17.180569,0,9.584022,0,20.285112,0,16.659315,0,21.912525,0,20.541174,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.253902546312031,-6.099122235717071,-5.375047515548679,
 -8.650033803260031];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.512878,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[13.318360,0,11.300838,0,11.183412,0,15.128844,0,19.105406,0,11.546366,0,21.431562,0,16.009203,0,21.034296,0,19.370539,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.701243948534571, 0.591593671655724,
 -0.583790984360805];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "159.428833,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.624201,0,9.180052,0,12.084344,0,14.725523,0,19.648396,0,11.378927,0,20.144154,0,16.795376,0,23.962957,0,20.361359,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.051084587221395,-1.7619898790632  ,-0.525658781921218,
  0.654953330273394];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "157.905289,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.728099,0,9.134651,0,9.141448,0,14.446856,0,18.822939,0,10.710155,0,19.582613,0,16.342466,0,21.545172,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.823233715440449,-3.596599767150177,-2.740306686080905,
 -5.21936497238845 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.452469,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.548172,0,8.102160,0,9.291077,0,13.294251,0,18.432613,0,10.150834,0,19.133327,0,16.514008,0,22.569891,0,19.750353,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  9.750589481736835, -7.782345006407989, -5.768037627853366,
 -11.146720738684836];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "146.786682,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.695583,0,8.330488,0,9.456502,0,14.503538,0,18.832407,0,10.872705,0,19.266920,0,15.643665,0,23.267664,0,19.012743,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.035834406802204,-2.960617237076384,-0.057953649720492,
 -2.431965663750205];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.882217,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.644466,0,9.321551,0,10.162717,0,12.639587,0,17.112204,0,10.195136,0,20.184792,0,16.100161,0,21.839153,0,19.091640,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  8.773499599839862, -7.969034383239045, -5.7655342327728  ,
 -10.169630856787863];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "147.291412,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.762109,0,9.324010,0,11.231086,0,15.140587,0,20.157898,0,12.751983,0,20.406006,0,16.994089,0,24.553400,0,20.031084,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-3.510961601381551,-1.7619898790632  ,-1.985535796081374,
  2.11483034443355 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "160.352249,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.790610,0,9.041201,0,10.588341,0,13.844358,0,16.998108,0,10.098713,0,19.908972,0,16.454185,0,22.426867,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.729718319322285,-5.412562558593048,-3.283607314743014,
 -6.125849576270286];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.329224,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.459682,0,8.436802,0,9.636287,0,13.597081,0,18.474184,0,10.680511,0,19.828970,0,15.267752,0,22.818264,0,18.798063,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.98525821342629 ,-5.024001309813752,-1.329738448776217,
 -5.381389470374289];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "146.997604,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.506981,0,9.107410,0,10.567950,0,14.046923,0,19.871737,0,12.234359,0,20.515467,0,16.565336,0,23.797129,0,20.119158,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.824750015383016,-1.7619898790632  ,-1.299324210082839,
  1.428618758435015];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "157.332443,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.580264,0,8.539710,0,9.235935,0,14.173610,0,18.879524,0,10.590276,0,19.748459,0,16.204517,0,23.138519,0,18.460451,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.286811381259966,-2.60821741102658 ,-0.661330450228058,
 -4.103588481772912];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.551270,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.414902,0,8.760850,0,9.680514,0,14.446856,0,18.640478,0,10.742403,0,19.457401,0,15.764630,0,22.475946,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.542339435960469,-3.021829793395294,-0.735437225021335,
 -2.93847069290847 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.561859,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.915848,0,9.489913,0,9.680514,0,13.331286,0,15.812118,0,9.776868,0,20.410576,0,15.913713,0,22.059513,0,19.634758,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.27179156913918 ,-5.833723256530059,-4.197359870590183,
 -7.667922826087181];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "147.025116,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.477414,0,8.854301,0,9.793958,0,14.357656,0,19.233587,0,10.420558,0,19.758364,0,16.379642,0,22.108593,0,18.909990,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.942169300462422,-4.235143849791778,-2.512832473434421,
 -5.338300557410423];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.294067,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.441309,0,8.842572,0,9.475744,0,13.866262,0,19.369394,0,10.802016,0,19.833033,0,15.748968,0,22.992027,0,18.671459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 2.319204424807575,-3.686409168228728,-0.615531736573519,
 -3.715335681755576];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.042801,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.814466,0,10.235244,0,11.475203,0,15.358282,0,21.108768,0,12.803126,0,22.282890,0,17.682701,0,23.418619,0,21.741167,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.88304039292169 ,  5.345437264297904, -9.846282592485302,
  -4.23927714588473 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.920471,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.695565,0,10.301453,0,12.001920,0,15.122973,0,20.826532,0,13.509687,0,21.662172,0,17.303011,0,24.254211,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-11.653947794960263,  4.234420990799369, -8.218713393876904,
  -3.644813797496057];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "168.057220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.853334,0,10.145765,0,11.976072,0,16.465721,0,20.625826,0,12.520370,0,21.421402,0,17.936142,0,24.834740,0,20.847595,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-7.079085227544103, 4.663600182388724,-9.005041796864353,
 -3.716843777687319];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.626968,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.229466,0,10.074260,0,11.809211,0,15.956264,0,20.933239,0,13.305439,0,21.813036,0,17.791399,0,24.834740,0,21.919149,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.815093958900693,  7.092872834061088,-12.409882178972119,
  -5.17054869892428 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "169.666183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.255444,0,9.816421,0,11.585485,0,16.130600,0,21.393314,0,12.307980,0,21.510801,0,18.540585,0,24.859528,0,21.577866,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-14.698606043891175,  6.604319786880912, -8.672349021593945,
  -7.930975761942104];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "168.978012,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.874243,0,8.704667,0,9.652082,0,13.777514,0,18.822939,0,10.581481,0,19.407114,0,15.876537,0,22.157425,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.803915134813008,-4.938098142089141,-3.251385520614596,
 -6.200046391761009];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.852066,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.807076,0,10.766244,0,12.002207,0,15.832060,0,20.139883,0,10.523823,0,20.495911,0,18.379803,0,23.524466,0,21.100805,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-5.075400858954649, 4.586531853531302,-4.904601142565624,
 -7.663147774271205];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "164.572266,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.395665,0,12.084382,0,11.315808,0,16.708710,0,20.599728,0,10.329348,0,21.305082,0,17.853863,0,23.178925,0,21.284290,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -6.482583623863584,  9.059144234512749, -9.596733384653069,
 -11.91624029414665 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.055801,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.874243,0,8.629945,0,10.020843,0,13.955915,0,18.845804,0,10.227386,0,19.432259,0,15.652912,0,21.961605,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.819132915879902,-6.099464604796661,-3.820366691268893,
 -7.215264172827903];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.778778,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.999690,0,10.110012,0,12.087503,0,16.045012,0,20.801359,0,12.583241,0,21.395243,0,18.280352,0,24.834740,0,20.669615,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-8.502674609259826, 4.568701958511771,-9.042882964770214,
 -3.489206162027552];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.806763,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.947095,0,10.607719,0,11.905709,0,15.662015,0,19.931786,0,12.721036,0,22.830719,0,18.158066,0,23.774323,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-11.720353091285947,  9.871227478836479,-14.989887824724073,
  -8.147252342723107];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "168.918152,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.728099,0,8.910295,0,9.793958,0,13.978271,0,18.754574,0,11.032000,0,19.432259,0,16.361149,0,22.426867,0,19.816408,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 2.236119381334004,-3.15391082519461 ,-1.663790534652682,
 -3.632250638282005];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "151.233871,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.268757,0,9.022472,0,10.219582,0,14.112185,0,19.689737,0,9.648195,0,19.708076,0,16.100161,0,21.545172,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 8.329054943467792,-7.914051244200959,-5.277976821578952,
 -9.725186200415793];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.673859,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.619326,0,9.590740,0,10.493567,0,15.297761,0,20.456532,0,10.562913,0,19.662107,0,16.150925,0,24.237356,0,20.539339,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.963423544453961,-1.761989879063199, 2.742676943522488,
 -2.613382395170313];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "158.610565,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.501499,0,8.455152,0,9.963116,0,14.059116,0,18.782749,0,10.435217,0,19.828970,0,15.458729,0,22.483135,0,18.759533,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.452880970384298,-5.527438352050079,-1.73914034239194 ,
 -5.849012227332297];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "147.727219,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.072965,0,10.913986,0,11.681409,0,15.240176,0,20.408955,0,12.608325,0,22.008343,0,17.235641,0,23.774323,0,21.181538,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-9.22312828354669 , 5.950559830521049,-9.49053632325923 ,
 -5.805268547557092];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.125656,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.564434,0,8.308544,0,9.382118,0,14.521153,0,19.400112,0,10.885084,0,18.883669,0,16.239996,0,23.425560,0,19.286135,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.178031474033403,-2.92582052229367 ,-0.234947431734406,
 -3.229711910117638];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.896790,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.555776,0,8.556168,0,10.053869,0,16.198799,0,19.721840,0,12.920723,0,19.876209,0,16.334730,0,24.866962,0,19.460445,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115824,-1.7619898790632  , 0.652339602184352,
 -0.523045053832176];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "157.545517,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.510571,0,9.484049,0,10.918042,0,14.113992,0,19.949570,0,12.662400,0,20.876877,0,16.560431,0,23.679636,0,20.196222,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-3.69353396618868 ,-1.7619898790632  ,-2.168108160888503,
  2.297402709240679];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "158.951797,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.085011,0,9.571634,0,11.157850,0,13.830583,0,19.624146,0,11.999165,0,19.815254,0,15.638948,0,23.961966,0,21.460436,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.328320659276226,-1.761989879063199, 1.463842304782051,
 -1.334547756429875];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "158.144989,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.164429,0,8.779578,0,9.907400,0,13.442843,0,18.685978,0,10.645654,0,18.880367,0,15.391924,0,22.181965,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.143915287400898,-6.846982190551056,-3.710524390853854,
 -7.5400465443489  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.439667,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.229273,0,8.235336,0,9.999302,0,15.146233,0,19.841249,0,10.557701,0,18.267267,0,16.326805,0,23.551233,0,19.559528,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[ 0.331780956621844,-3.061971852721466, 0.596302742354399,
 -2.587192929571313];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.713928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.041294,0,9.041201,0,10.162717,0,14.446856,0,19.917698,0,11.385768,0,19.733221,0,16.715174,0,21.986141,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.601944194646503,-2.047004048236113,-0.822690795577975,
 -1.998075451594504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "154.789597,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.381102,0,10.480219,0,11.297141,0,16.224318,0,21.005527,0,13.134092,0,21.272827,0,17.978224,0,24.936369,0,21.976030,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.120474368170958,  8.014079308365288,-12.757927940050976,
  -6.664915886453822];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "169.685852,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.421669,0,8.920321,0,9.649498,0,14.327622,0,18.879524,0,10.276249,0,19.186663,0,15.721038,0,22.854206,0,18.829256,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.065162769700704,-4.509571263732208,-1.217369865097018,
 -4.461294026648703];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.066040,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.832427,0,8.891756,0,11.042110,0,14.558414,0,19.666872,0,9.937790,0,20.009293,0,16.920303,0,21.814613,0,19.091640,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.529948345567402,-5.991879651488556,-3.917089294113131,
 -6.926079602515403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "152.765228,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.581954,0,8.742124,0,9.623938,0,13.955915,0,17.408758,0,9.937790,0,19.858685,0,16.435692,0,22.279877,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.157389751817402,-5.398891637267364,-4.007189582504245,
 -7.553521008765403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.822800,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.695565,0,10.658795,0,11.777620,0,14.841823,0,20.379160,0,13.059170,0,22.008343,0,17.888208,0,24.455984,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.711195910499814,  7.381609177444877,-12.42839395112788 ,
  -5.729509613536097];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "168.144363,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.749006,0,8.387051,0,9.538640,0,13.978271,0,19.028263,0,9.326349,0,19.256506,0,15.857855,0,21.839153,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  9.318592594500263, -8.163409421386017, -5.950563674576753,
 -11.706036961858912];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "147.959152,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.605597,0,10.235244,0,11.642063,0,15.867740,0,21.306242,0,12.206345,0,21.734556,0,19.022179,0,24.492174,0,21.741167,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-16.89208367450641 ,  8.969080185745659,-11.29920487181367 ,
 -10.033640709451868];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "169.853317,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.392304,0,9.180052,0,10.946762,0,14.419757,0,19.486261,0,11.886454,0,20.353685,0,16.777260,0,23.986753,0,19.865948,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-1.630144188051473,-1.7619898790632  ,-0.104718382751296,
  0.234012931103472];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "157.295242,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.498322,0,9.059929,0,10.162717,0,13.018746,0,18.959667,0,10.356059,0,19.081009,0,16.230562,0,22.696308,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.485362163926777,-6.686109731139434,-4.466867278732273,
 -7.881493420874778];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.061386,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.046777,0,9.859363,0,10.551292,0,14.643098,0,19.023182,0,11.085097,0,19.789604,0,16.226786,0,22.860403,0,20.355854,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.761989879063199, 0.652339602184352,
 -0.523045053832177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "156.441437,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.611723,0,10.224838,0,11.777620,0,16.458496,0,21.333727,0,12.833747,0,21.921738,0,18.113153,0,24.380381,0,21.777866,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-13.349480181769593,  7.929915642593802,-11.747147605578318,
  -7.50736888938351 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "170.433289,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.068108,0,9.782558,0,10.349107,0,14.981833,0,19.208181,0,11.019946,0,20.836243,0,16.416252,0,22.760014,0,20.724659,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.761989879063199, 0.652339602184352,
 -0.523045053832176];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "158.146896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.313098,0,10.020535,0,11.447344,0,16.399330,0,21.108768,0,12.363358,0,21.264698,0,18.316586,0,25.077656,0,21.561354,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.44121888263629 ,  7.144162392471732, -9.378528640429394,
  -8.304481354288294];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "168.872726,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.555766,0,9.816231,0,11.648669,0,14.564511,0,20.288855,0,12.638294,0,20.529438,0,16.886335,0,24.160763,0,19.409071,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.70431334698702 ,-1.7619898790632  ,-1.178887541686843,
  1.308182090039018];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "160.497940,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.464945,0,10.378067,0,11.873830,0,15.778992,0,20.856558,0,12.946458,0,21.705349,0,17.550604,0,24.153080,0,20.981539,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-8.765935862587217, 4.135348534439506,-8.513126006791945,
 -3.152256271861291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.689423,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.270860,0,9.527180,0,10.304592,0,13.375773,0,16.336632,0,10.710155,0,20.385431,0,17.237152,0,22.010681,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.40400420799416 ,-4.232783505858673,-3.650792907394381,
 -5.800135464942161];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "151.336334,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.987028,0,9.128598,0,10.758074,0,14.251744,0,19.700132,0,11.654515,0,19.707569,0,16.061853,0,24.086151,0,20.119158,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 3 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.077053138857137,-1.7619898790632  , 0.44837266644304 ,
 -0.319078118090864];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "155.454819,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[12.219531,0,10.939524,0,11.681409,0,15.521326,0,20.438747,0,13.171880,0,22.181555,0,18.630602,0,24.052195,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-13.439249957130674, 10.634545540665092,-16.03865396089351 ,
  -8.6251223302109  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "170.216476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.295376,0,8.349405,0,9.318647,0,13.294251,0,19.185316,0,9.711390,0,18.999483,0,15.837097,0,22.879986,0,19.381548,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  9.262123630877213, -8.208008000792756, -5.144672637589448,
 -10.841756261175314];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "146.252502,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.323043,0,8.364161,0,8.880674,0,15.406383,0,19.630381,0,10.766835,0,19.055864,0,15.434008,0,25.156979,0,18.087791,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.67924385695396 ,-1.7619898790632  , 2.458497256022488,
 -2.329202707670313];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.106110,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.298737,0,9.763453,0,11.157850,0,15.049355,0,20.179148,0,11.672759,0,19.457911,0,16.605909,0,23.586433,0,20.539339,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.943495111474367,-1.761989879063199, 0.722748510542894,
 -0.593453962190719];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "159.310883,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.996303,0,11.933614,0,11.573135,0,16.326614,0,20.093922,0,12.015778,0,20.344288,0,17.571926,0,23.795889,0,21.465940,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-5.284723159367834, 7.194359993790092,-8.672458369435155,
 -9.110946827919252];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "167.117401,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.379642,0,8.882109,0,9.208364,0,13.777966,0,18.714848,0,10.496134,0,19.828716,0,15.295681,0,22.414968,0,19.381548,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.423229328538594,-5.044640729369415,-1.726188491500827,
 -5.819360585486595];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "147.379990,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[9.472777,0,8.716395,0,8.951612,0,14.808852,0,19.402420,0,11.100734,0,20.198759,0,15.773688,0,24.305027,0,20.128332,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.331677252339703,-1.7619898790632  , 1.110930651408231,
 -0.981636103056055];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "152.858597,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[11.061991,0,8.629945,0,10.020843,0,13.732801,0,19.210724,0,10.002289,0,19.808397,0,16.528913,0,21.716702,0,19.634758,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 8.07191805684157 ,-6.921806523741973,-5.567785506852144,
 -9.74539414722268 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "150.347351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.113109,0,8.464799,0,10.663874,0,14.383626,0,19.072376,0,9.995774,0,18.621819,0,16.057890,0,24.980986,0,18.961365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.890465535254192,-2.117943112261992, 1.66971893432272 ,
 -2.252330852368129];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "151.315613,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.498322,0,9.153378,0,9.623938,0,13.465199,0,16.747282,0,10.259635,0,19.582613,0,16.845762,0,22.769680,0,19.816408,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.141856304551777,-4.70547504180838 ,-3.640211890853854,
 -6.537987561499778];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "148.762222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.593569,0,9.725051,0,11.186571,0,16.426430,0,20.340820,0,11.574706,0,20.274698,0,16.966162,0,23.886858,0,19.803564,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.7619898790632  , 0.652339602184352,
 -0.523045053832177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "160.778427,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[10.749006,0,8.929023,0,9.595505,0,12.885058,0,17.271797,0,9.970040,0,19.883829,0,16.957479,0,22.500238,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 8.567916981126483,-6.627852628173126,-5.953423331893894,
 -9.964048238074485];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUS_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUS_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUS_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUS_convergence_times.csv", ","];
  ];
  WriteString["VIRUS_convergence_times.csv", "149.101501,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
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
rsys=VIRUSReduced[11.461777,0,9.516586,0,11.103571,0,14.458374,0,19.159218,0,10.870099,0,19.596327,0,16.100349,0,23.158102,0,19.741179,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["0: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.178133221687052,-2.456942860491973,-0.042613379244421,
 -1.217998035260949];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "155.165588,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.689230,0,10.593153,0,12.171078,0,15.889871,0,20.779415,0,12.614514,0,22.152349,0,18.841770,0,24.272554,0,21.741167,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-14.931818514899476, 10.43665144429058 ,-15.212276504198924,
  -9.055711594156458];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "170.745102,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.870010,0,11.199254,0,11.230225,0,16.371553,0,20.645689,0,10.977923,0,20.521309,0,18.304695,0,23.573792,0,21.831076,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-7.607293548191322, 7.071883416031302,-8.159970416219576,
 -9.378481625617253];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "166.525528,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.611319,0,8.482581,0,9.677068,0,13.206181,0,17.961912,0,10.433263,0,19.775127,0,15.779161,0,22.363163,0,19.381548,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["3: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.587063900376973,-7.107708165587677,-4.03571016661069 ,
 -8.983195157324973];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "146.671326,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.046584,0,8.246875,0,8.809737,0,14.268229,0,19.272160,0,11.980596,0,19.027927,0,15.700846,0,24.358320,0,18.898981,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["4: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.487207228230816,-1.7619898790632  , 1.266460627299344,
 -1.137166078947168];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.610260,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.941411,0,8.693128,0,9.600100,0,13.298091,0,18.385958,0,10.230643,0,19.808397,0,15.128860,0,22.273928,0,18.348526,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.752140156175313,-6.856720158995881,-3.541480849899263,
 -8.148271413123314];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "145.709045,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.211110,0,8.406535,0,9.704639,0,13.513978,0,19.091314,0,10.056690,0,18.785633,0,16.185270,0,23.009378,0,19.750353,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["6: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.802646747972187,-7.446148106993927,-4.352478812850925,
 -9.198778004920188];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "147.714905,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.519230,0,9.078656,0,9.708947,0,13.598888,0,19.233587,0,10.516981,0,19.658043,0,16.267925,0,21.594252,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.970306507493672,-6.337656209410919,-4.630453895248386,
 -8.366437764441672];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.354385,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.025446,0,9.686650,0,10.435553,0,14.981833,0,19.092472,0,11.411503,0,20.989389,0,16.056190,0,22.409515,0,19.987049,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["8: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.761989879063199, 0.652339602184352,
 -0.523045053832177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "157.075592,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.581954,0,8.835573,0,10.389889,0,13.933559,0,18.572113,0,9.487272,0,19.131042,0,16.267925,0,22.451406,0,19.453108,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.434397808458027,-7.401723096312774,-5.076303313888523,
 -8.830529065406028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.103836,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.362940,0,9.840067,0,11.157850,0,14.665680,0,20.248436,0,11.183149,0,20.402449,0,16.605909,0,23.361115,0,20.539339,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.534220851536968,-1.761989879063199, 1.013126286105293,
 -0.883831737753117];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "159.366928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.170544,0,9.283906,0,11.619949,0,14.620515,0,19.554855,0,11.117999,0,18.845320,0,15.601018,0,24.763102,0,20.355854,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.378010110864015,-1.761989879063199, 1.157263509932543,
 -1.027968961580367];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "156.933060,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.372672,0,8.752905,0,9.818369,0,13.924300,0,19.147902,0,10.324461,0,19.361906,0,15.514776,0,22.375805,0,18.247608,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.568019977952657,-6.386634061278595,-2.67030317655942 ,
 -6.964151234900656];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "146.840698,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.332519,0,11.519520,0,11.087201,0,15.247402,0,19.449999,0,10.913099,0,20.976437,0,17.816309,0,23.376234,0,21.649426,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-5.903618690068518, 7.884038185928763,-9.41854830473789 ,
 -9.744213276893861];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "164.368149,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.768849,0,10.242999,0,10.984385,0,14.643098,0,20.410109,0,11.999165,0,20.734142,0,16.321520,0,23.686577,0,19.987049,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["14: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.112634727602254,-1.7619898790632  , 0.412791077697922,
 -0.283496529345747];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "160.777878,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.689230,0,10.235244,0,11.809211,0,16.886658,0,21.152651,0,12.206345,0,21.578108,0,18.841770,0,24.492174,0,22.276945,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["15: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-15.313951045111878, 10.706696724747124,-12.783066795031443,
 -12.025011864237024];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "171.168335,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.936333,0,9.909870,0,11.491286,0,15.199302,0,19.969896,0,12.336646,0,21.738873,0,17.085239,0,22.417200,0,19.739344,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.80696112835909 ,-1.7619898790632  ,-1.281535323058913,
  1.410829871411089];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "160.823990,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.555123,0,9.667543,0,10.233367,0,14.123928,0,20.109627,0,11.509556,0,20.453497,0,17.288481,0,23.561396,0,21.091629,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-5.683229952887753,-1.209091663362073, 0.410188367064569,
 -1.386690250114646];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "159.594147,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.399678,0,13.420491,0,12.671948,0,17.706398,0,21.415947,0,13.054283,0,23.535759,0,16.940119,0,22.637810,0,20.133837,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -9.04938876300553 , 11.171586251114311,-14.454616267384374,
 -11.283241444618472];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "173.916275,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.337403,0,9.167566,0,10.311484,0,11.733131,0,18.197262,0,11.312148,0,19.748459,0,14.193225,0,22.544113,0,18.091097,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.171382061387716,-8.184150884093537,-2.61788446775815 ,
 -7.567513318335717];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "144.635895,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.728099,0,8.873028,0,9.680514,0,14.781528,0,19.210724,0,9.905540,0,20.034182,0,16.267925,0,21.692162,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.789230457688984,-4.591627309264434,-3.121726821578952,
 -6.185361714636985];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "151.171768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.521747,0,11.594810,0,11.687728,0,15.877000,0,20.001999,0,11.464275,0,20.445372,0,17.628351,0,22.413731,0,21.831076,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-4.694120371864559, 5.490919327591361,-6.493525138566846,
 -7.8829987263901  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "165.466080,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.195474,0,9.308120,0,10.889609,0,15.293469,0,20.644304,0,11.647675,0,20.876877,0,16.488153,0,23.868517,0,20.361359,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["22: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.602022239809285,-1.7619898790632  ,-1.076596434509109,
  1.205890982861284];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "159.573547,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.140766,0,9.547988,0,11.032059,0,14.070408,0,19.370548,0,12.244457,0,20.667601,0,16.668751,0,23.419861,0,20.031084,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-1.843864509706747,-1.761989879063199,-0.31843870440657 ,
  0.447733252758745];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "158.193527,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.021663,0,8.364161,0,10.300283,0,15.292567,0,19.955807,0,11.737909,0,19.892210,0,16.016184,0,23.160334,0,20.128332,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["24: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115824,-1.7619898790632  , 0.652339602184352,
 -0.523045053832176];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "154.869446,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.611319,0,8.254252,0,9.649498,0,13.580144,0,19.279549,0,10.527406,0,19.694870,0,15.160186,0,22.906012,0,19.935673,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.472138515855488,-5.36681480163504 ,-1.70341093412778 ,
 -5.868269772803489];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.598892,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.519230,0,8.854301,0,9.964265,0,13.822001,0,16.405228,0,9.873616,0,19.457401,0,16.715174,0,23.112494,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["26: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.772935978318867,-4.821990201415313,-3.306718658126803,
 -6.169067235266868];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.083237,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.683317,0,9.610036,0,10.522286,0,13.898331,0,19.693665,0,11.672759,0,20.836243,0,16.188856,0,23.160830,0,20.908146,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.797807849522808,-1.761989879063199, 0.741997631625312,
 -0.612703083273137];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "158.174469,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.087977,0,8.711477,0,9.708947,0,14.086216,0,18.672581,0,10.189924,0,19.746681,0,15.128860,0,23.090681,0,18.583387,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.93811236992043 ,-3.734049031676544,-0.456528495468112,
 -3.334243626868431];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.006729,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.108665,0,10.071800,0,11.585485,0,16.458496,0,21.244343,0,12.796286,0,21.013008,0,18.765720,0,25.263565,0,20.783375,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-11.251672210300697,  5.972169136856498, -9.114759577901706,
  -6.224263905585514];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "169.090744,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.710974,0,11.688827,0,11.859183,0,15.741957,0,20.254902,0,10.556398,0,21.355621,0,17.741013,0,23.104811,0,21.465940,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["30: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -6.564314806984189,  8.367173409317436, -9.028920760820265,
 -11.10011126758883 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "166.479630,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.554701,0,10.992113,0,11.659007,0,17.518061,0,20.645689,0,11.204974,0,19.762682,0,18.060503,0,25.030560,0,21.100805,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -6.018456083275204,  6.620846962784232, -6.021720223947503,
 -10.614658911395185];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.529083,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.479728,0,9.059929,0,10.077707,0,13.732801,0,17.180569,0,9.584022,0,20.285112,0,16.659315,0,21.912525,0,20.541174,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["32: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 7.253902546312031,-6.099122235717071,-5.375047515548679,
 -8.650033803260031];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.512878,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[13.318360,0,11.300838,0,11.183412,0,15.128844,0,19.105406,0,11.546366,0,21.431562,0,16.009203,0,21.034296,0,19.370539,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["33: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.701243948534571, 0.591593671655724,
 -0.583790984360805];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "159.428833,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.624201,0,9.180052,0,12.084344,0,14.725523,0,19.648396,0,11.378927,0,20.144154,0,16.795376,0,23.962957,0,20.361359,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["34: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.051084587221395,-1.7619898790632  ,-0.525658781921218,
  0.654953330273394];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "157.905289,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.728099,0,9.134651,0,9.141448,0,14.446856,0,18.822939,0,10.710155,0,19.582613,0,16.342466,0,21.545172,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.823233715440449,-3.596599767150177,-2.740306686080905,
 -5.21936497238845 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.452469,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.548172,0,8.102160,0,9.291077,0,13.294251,0,18.432613,0,10.150834,0,19.133327,0,16.514008,0,22.569891,0,19.750353,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  9.750589481736835, -7.782345006407989, -5.768037627853366,
 -11.146720738684836];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "146.786682,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.695583,0,8.330488,0,9.456502,0,14.503538,0,18.832407,0,10.872705,0,19.266920,0,15.643665,0,23.267664,0,19.012743,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["37: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.035834406802204,-2.960617237076384,-0.057953649720492,
 -2.431965663750205];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.882217,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.644466,0,9.321551,0,10.162717,0,12.639587,0,17.112204,0,10.195136,0,20.184792,0,16.100161,0,21.839153,0,19.091640,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["38: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  8.773499599839862, -7.969034383239045, -5.7655342327728  ,
 -10.169630856787863];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "147.291412,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.762109,0,9.324010,0,11.231086,0,15.140587,0,20.157898,0,12.751983,0,20.406006,0,16.994089,0,24.553400,0,20.031084,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-3.510961601381551,-1.7619898790632  ,-1.985535796081374,
  2.11483034443355 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "160.352249,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.790610,0,9.041201,0,10.588341,0,13.844358,0,16.998108,0,10.098713,0,19.908972,0,16.454185,0,22.426867,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.729718319322285,-5.412562558593048,-3.283607314743014,
 -6.125849576270286];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.329224,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.459682,0,8.436802,0,9.636287,0,13.597081,0,18.474184,0,10.680511,0,19.828970,0,15.267752,0,22.818264,0,18.798063,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.98525821342629 ,-5.024001309813752,-1.329738448776217,
 -5.381389470374289];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "146.997604,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.506981,0,9.107410,0,10.567950,0,14.046923,0,19.871737,0,12.234359,0,20.515467,0,16.565336,0,23.797129,0,20.119158,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["42: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.824750015383016,-1.7619898790632  ,-1.299324210082839,
  1.428618758435015];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "157.332443,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.580264,0,8.539710,0,9.235935,0,14.173610,0,18.879524,0,10.590276,0,19.748459,0,16.204517,0,23.138519,0,18.460451,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["43: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.286811381259966,-2.60821741102658 ,-0.661330450228058,
 -4.103588481772912];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.551270,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.414902,0,8.760850,0,9.680514,0,14.446856,0,18.640478,0,10.742403,0,19.457401,0,15.764630,0,22.475946,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.542339435960469,-3.021829793395294,-0.735437225021335,
 -2.93847069290847 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.561859,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.915848,0,9.489913,0,9.680514,0,13.331286,0,15.812118,0,9.776868,0,20.410576,0,15.913713,0,22.059513,0,19.634758,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.27179156913918 ,-5.833723256530059,-4.197359870590183,
 -7.667922826087181];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "147.025116,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.477414,0,8.854301,0,9.793958,0,14.357656,0,19.233587,0,10.420558,0,19.758364,0,16.379642,0,22.108593,0,18.909990,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.942169300462422,-4.235143849791778,-2.512832473434421,
 -5.338300557410423];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.294067,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.441309,0,8.842572,0,9.475744,0,13.866262,0,19.369394,0,10.802016,0,19.833033,0,15.748968,0,22.992027,0,18.671459,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 2.319204424807575,-3.686409168228728,-0.615531736573519,
 -3.715335681755576];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.042801,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.814466,0,10.235244,0,11.475203,0,15.358282,0,21.108768,0,12.803126,0,22.282890,0,17.682701,0,23.418619,0,21.741167,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["48: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.88304039292169 ,  5.345437264297904, -9.846282592485302,
  -4.23927714588473 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.920471,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.695565,0,10.301453,0,12.001920,0,15.122973,0,20.826532,0,13.509687,0,21.662172,0,17.303011,0,24.254211,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-11.653947794960263,  4.234420990799369, -8.218713393876904,
  -3.644813797496057];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "168.057220,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.853334,0,10.145765,0,11.976072,0,16.465721,0,20.625826,0,12.520370,0,21.421402,0,17.936142,0,24.834740,0,20.847595,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["50: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-7.079085227544103, 4.663600182388724,-9.005041796864353,
 -3.716843777687319];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.626968,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.229466,0,10.074260,0,11.809211,0,15.956264,0,20.933239,0,13.305439,0,21.813036,0,17.791399,0,24.834740,0,21.919149,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.815093958900693,  7.092872834061088,-12.409882178972119,
  -5.17054869892428 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "169.666183,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.255444,0,9.816421,0,11.585485,0,16.130600,0,21.393314,0,12.307980,0,21.510801,0,18.540585,0,24.859528,0,21.577866,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["52: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-14.698606043891175,  6.604319786880912, -8.672349021593945,
  -7.930975761942104];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "168.978012,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.874243,0,8.704667,0,9.652082,0,13.777514,0,18.822939,0,10.581481,0,19.407114,0,15.876537,0,22.157425,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.803915134813008,-4.938098142089141,-3.251385520614596,
 -6.200046391761009];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.852066,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.807076,0,10.766244,0,12.002207,0,15.832060,0,20.139883,0,10.523823,0,20.495911,0,18.379803,0,23.524466,0,21.100805,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-5.075400858954649, 4.586531853531302,-4.904601142565624,
 -7.663147774271205];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "164.572266,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.395665,0,12.084382,0,11.315808,0,16.708710,0,20.599728,0,10.329348,0,21.305082,0,17.853863,0,23.178925,0,21.284290,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[ -6.482583623863584,  9.059144234512749, -9.596733384653069,
 -11.91624029414665 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.055801,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.874243,0,8.629945,0,10.020843,0,13.955915,0,18.845804,0,10.227386,0,19.432259,0,15.652912,0,21.961605,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["56: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.819132915879902,-6.099464604796661,-3.820366691268893,
 -7.215264172827903];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.778778,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.999690,0,10.110012,0,12.087503,0,16.045012,0,20.801359,0,12.583241,0,21.395243,0,18.280352,0,24.834740,0,20.669615,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["57: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-8.502674609259826, 4.568701958511771,-9.042882964770214,
 -3.489206162027552];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.806763,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.947095,0,10.607719,0,11.905709,0,15.662015,0,19.931786,0,12.721036,0,22.830719,0,18.158066,0,23.774323,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-11.720353091285947,  9.871227478836479,-14.989887824724073,
  -8.147252342723107];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "168.918152,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.728099,0,8.910295,0,9.793958,0,13.978271,0,18.754574,0,11.032000,0,19.432259,0,16.361149,0,22.426867,0,19.816408,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 2.236119381334004,-3.15391082519461 ,-1.663790534652682,
 -3.632250638282005];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "151.233871,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.268757,0,9.022472,0,10.219582,0,14.112185,0,19.689737,0,9.648195,0,19.708076,0,16.100161,0,21.545172,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 8.329054943467792,-7.914051244200959,-5.277976821578952,
 -9.725186200415793];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.673859,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.619326,0,9.590740,0,10.493567,0,15.297761,0,20.456532,0,10.562913,0,19.662107,0,16.150925,0,24.237356,0,20.539339,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.963423544453961,-1.761989879063199, 2.742676943522488,
 -2.613382395170313];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "158.610565,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.501499,0,8.455152,0,9.963116,0,14.059116,0,18.782749,0,10.435217,0,19.828970,0,15.458729,0,22.483135,0,18.759533,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["62: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.452880970384298,-5.527438352050079,-1.73914034239194 ,
 -5.849012227332297];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "147.727219,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.072965,0,10.913986,0,11.681409,0,15.240176,0,20.408955,0,12.608325,0,22.008343,0,17.235641,0,23.774323,0,21.181538,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["63: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-9.22312828354669 , 5.950559830521049,-9.49053632325923 ,
 -5.805268547557092];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.125656,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.564434,0,8.308544,0,9.382118,0,14.521153,0,19.400112,0,10.885084,0,18.883669,0,16.239996,0,23.425560,0,19.286135,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["64: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.178031474033403,-2.92582052229367 ,-0.234947431734406,
 -3.229711910117638];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.896790,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.555776,0,8.556168,0,10.053869,0,16.198799,0,19.721840,0,12.920723,0,19.876209,0,16.334730,0,24.866962,0,19.460445,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115824,-1.7619898790632  , 0.652339602184352,
 -0.523045053832176];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "157.545517,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.510571,0,9.484049,0,10.918042,0,14.113992,0,19.949570,0,12.662400,0,20.876877,0,16.560431,0,23.679636,0,20.196222,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["66: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-3.69353396618868 ,-1.7619898790632  ,-2.168108160888503,
  2.297402709240679];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "158.951797,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.085011,0,9.571634,0,11.157850,0,13.830583,0,19.624146,0,11.999165,0,19.815254,0,15.638948,0,23.961966,0,21.460436,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["67: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-3.328320659276226,-1.761989879063199, 1.463842304782051,
 -1.334547756429875];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "158.144989,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.164429,0,8.779578,0,9.907400,0,13.442843,0,18.685978,0,10.645654,0,18.880367,0,15.391924,0,22.181965,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["68: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.143915287400898,-6.846982190551056,-3.710524390853854,
 -7.5400465443489  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.439667,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.229273,0,8.235336,0,9.999302,0,15.146233,0,19.841249,0,10.557701,0,18.267267,0,16.326805,0,23.551233,0,19.559528,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 0 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[ 0.331780956621844,-3.061971852721466, 0.596302742354399,
 -2.587192929571313];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.713928,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.041294,0,9.041201,0,10.162717,0,14.446856,0,19.917698,0,11.385768,0,19.733221,0,16.715174,0,21.986141,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.601944194646503,-2.047004048236113,-0.822690795577975,
 -1.998075451594504];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "154.789597,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.381102,0,10.480219,0,11.297141,0,16.224318,0,21.005527,0,13.134092,0,21.272827,0,17.978224,0,24.936369,0,21.976030,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["71: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.120474368170958,  8.014079308365288,-12.757927940050976,
  -6.664915886453822];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "169.685852,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.421669,0,8.920321,0,9.649498,0,14.327622,0,18.879524,0,10.276249,0,19.186663,0,15.721038,0,22.854206,0,18.829256,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.065162769700704,-4.509571263732208,-1.217369865097018,
 -4.461294026648703];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.066040,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.832427,0,8.891756,0,11.042110,0,14.558414,0,19.666872,0,9.937790,0,20.009293,0,16.920303,0,21.814613,0,19.091640,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.529948345567402,-5.991879651488556,-3.917089294113131,
 -6.926079602515403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "152.765228,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.581954,0,8.742124,0,9.623938,0,13.955915,0,17.408758,0,9.937790,0,19.858685,0,16.435692,0,22.279877,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.157389751817402,-5.398891637267364,-4.007189582504245,
 -7.553521008765403];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.822800,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.695565,0,10.658795,0,11.777620,0,14.841823,0,20.379160,0,13.059170,0,22.008343,0,17.888208,0,24.455984,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.711195910499814,  7.381609177444877,-12.42839395112788 ,
  -5.729509613536097];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "168.144363,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.749006,0,8.387051,0,9.538640,0,13.978271,0,19.028263,0,9.326349,0,19.256506,0,15.857855,0,21.839153,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  9.318592594500263, -8.163409421386017, -5.950563674576753,
 -11.706036961858912];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "147.959152,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.605597,0,10.235244,0,11.642063,0,15.867740,0,21.306242,0,12.206345,0,21.734556,0,19.022179,0,24.492174,0,21.741167,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["77: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-16.89208367450641 ,  8.969080185745659,-11.29920487181367 ,
 -10.033640709451868];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "169.853317,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.392304,0,9.180052,0,10.946762,0,14.419757,0,19.486261,0,11.886454,0,20.353685,0,16.777260,0,23.986753,0,19.865948,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-1.630144188051473,-1.7619898790632  ,-0.104718382751296,
  0.234012931103472];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "157.295242,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.498322,0,9.059929,0,10.162717,0,13.018746,0,18.959667,0,10.356059,0,19.081009,0,16.230562,0,22.696308,0,19.998058,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 6.485362163926777,-6.686109731139434,-4.466867278732273,
 -7.881493420874778];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.061386,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.046777,0,9.859363,0,10.551292,0,14.643098,0,19.023182,0,11.085097,0,19.789604,0,16.226786,0,22.860403,0,20.355854,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.761989879063199, 0.652339602184352,
 -0.523045053832177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "156.441437,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.611723,0,10.224838,0,11.777620,0,16.458496,0,21.333727,0,12.833747,0,21.921738,0,18.113153,0,24.380381,0,21.777866,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-13.349480181769593,  7.929915642593802,-11.747147605578318,
  -7.50736888938351 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "170.433289,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.068108,0,9.782558,0,10.349107,0,14.981833,0,19.208181,0,11.019946,0,20.836243,0,16.416252,0,22.760014,0,20.724659,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["82: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.761989879063199, 0.652339602184352,
 -0.523045053832176];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "158.146896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.313098,0,10.020535,0,11.447344,0,16.399330,0,21.108768,0,12.363358,0,21.264698,0,18.316586,0,25.077656,0,21.561354,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["83: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-12.44121888263629 ,  7.144162392471732, -9.378528640429394,
  -8.304481354288294];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "168.872726,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.555766,0,9.816231,0,11.648669,0,14.564511,0,20.288855,0,12.638294,0,20.529438,0,16.886335,0,24.160763,0,19.409071,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 3;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 3 " <> ToString[predicted] <> " 3 " <> warning];
correctResults=List[-2.70431334698702 ,-1.7619898790632  ,-1.178887541686843,
  1.308182090039018];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "160.497940,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.464945,0,10.378067,0,11.873830,0,15.778992,0,20.856558,0,12.946458,0,21.705349,0,17.550604,0,24.153080,0,20.981539,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-8.765935862587217, 4.135348534439506,-8.513126006791945,
 -3.152256271861291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.689423,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.270860,0,9.527180,0,10.304592,0,13.375773,0,16.336632,0,10.710155,0,20.385431,0,17.237152,0,22.010681,0,20.177874,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["86: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.40400420799416 ,-4.232783505858673,-3.650792907394381,
 -5.800135464942161];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "151.336334,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.987028,0,9.128598,0,10.758074,0,14.251744,0,19.700132,0,11.654515,0,19.707569,0,16.061853,0,24.086151,0,20.119158,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 3;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 3 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.077053138857137,-1.7619898790632  , 0.44837266644304 ,
 -0.319078118090864];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "155.454819,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[12.219531,0,10.939524,0,11.681409,0,15.521326,0,20.438747,0,13.171880,0,22.181555,0,18.630602,0,24.052195,0,21.379704,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-13.439249957130674, 10.634545540665092,-16.03865396089351 ,
  -8.6251223302109  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "170.216476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.295376,0,8.349405,0,9.318647,0,13.294251,0,19.185316,0,9.711390,0,18.999483,0,15.837097,0,22.879986,0,19.381548,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[  9.262123630877213, -8.208008000792756, -5.144672637589448,
 -10.841756261175314];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "146.252502,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.323043,0,8.364161,0,8.880674,0,15.406383,0,19.630381,0,10.766835,0,19.055864,0,15.434008,0,25.156979,0,18.087791,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["90: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-2.67924385695396 ,-1.7619898790632  , 2.458497256022488,
 -2.329202707670313];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.106110,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.298737,0,9.763453,0,11.157850,0,15.049355,0,20.179148,0,11.672759,0,19.457911,0,16.605909,0,23.586433,0,20.539339,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.943495111474367,-1.761989879063199, 0.722748510542894,
 -0.593453962190719];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "159.310883,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.996303,0,11.933614,0,11.573135,0,16.326614,0,20.093922,0,12.015778,0,20.344288,0,17.571926,0,23.795889,0,21.465940,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-5.284723159367834, 7.194359993790092,-8.672458369435155,
 -9.110946827919252];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "167.117401,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.379642,0,8.882109,0,9.208364,0,13.777966,0,18.714848,0,10.496134,0,19.828716,0,15.295681,0,22.414968,0,19.381548,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 4.423229328538594,-5.044640729369415,-1.726188491500827,
 -5.819360585486595];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "147.379990,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[9.472777,0,8.716395,0,8.951612,0,14.808852,0,19.402420,0,11.100734,0,20.198759,0,15.773688,0,24.305027,0,20.128332,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["94: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.331677252339703,-1.7619898790632  , 1.110930651408231,
 -0.981636103056055];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "152.858597,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[11.061991,0,8.629945,0,10.020843,0,13.732801,0,19.210724,0,10.002289,0,19.808397,0,16.528913,0,21.716702,0,19.634758,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 8.07191805684157 ,-6.921806523741973,-5.567785506852144,
 -9.74539414722268 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "150.347351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.113109,0,8.464799,0,10.663874,0,14.383626,0,19.072376,0,9.995774,0,18.621819,0,16.057890,0,24.980986,0,18.961365,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-1.890465535254192,-2.117943112261992, 1.66971893432272 ,
 -2.252330852368129];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "151.315613,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.498322,0,9.153378,0,9.623938,0,13.465199,0,16.747282,0,10.259635,0,19.582613,0,16.845762,0,22.769680,0,19.816408,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["97: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 5.141856304551777,-4.70547504180838 ,-3.640211890853854,
 -6.537987561499778];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "148.762222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.593569,0,9.725051,0,11.186571,0,16.426430,0,20.340820,0,11.574706,0,20.274698,0,16.966162,0,23.886858,0,19.803564,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 2;
nnOutput = 2;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["98: 2 " <> ToString[predicted] <> " 2 " <> warning];
correctResults=List[-0.873086203115825,-1.7619898790632  , 0.652339602184352,
 -0.523045053832177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "160.778427,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[10.749006,0,8.929023,0,9.595505,0,12.885058,0,17.271797,0,9.970040,0,19.883829,0,16.957479,0,22.500238,0,20.359524,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol,(hL1N3Spos[tmax] - hL1N3Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["99: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 8.567916981126483,-6.627852628173126,-5.953423331893894,
 -9.964048238074485];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus-inputs-shifted", "VIRUSReduced_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["VIRUSReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["VIRUSReduced_convergence_times.csv", ","];
  ];
  WriteString["VIRUSReduced_convergence_times.csv", "149.101501,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
