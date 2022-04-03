#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

VIRUS[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_] :=
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
rxn[x0Spos,iL0N1Spos+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Spos,1],
rxn[x0Sneg,iL0N1Sneg+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N5Sneg+iL0N6Sneg,1],
rxn[x1Spos,iL0N0Sneg+iL0N1Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Sneg+iL0N6Spos,1],
rxn[x1Sneg,iL0N0Spos+iL0N1Spos+iL0N3Spos+iL0N4Sneg+iL0N5Spos+iL0N6Sneg,1],
rxn[x2Spos,iL0N0Sneg+iL0N1Spos+iL0N5Sneg+iL0N7Spos,1],
rxn[x2Sneg,iL0N0Spos+iL0N1Sneg+iL0N5Spos+iL0N7Sneg,1],
rxn[x3Spos,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N7Sneg,1],
rxn[x3Sneg,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N7Spos,1],
rxn[x4Spos,iL0N1Spos+iL0N2Spos+iL0N4Sneg+iL0N5Spos,1],
rxn[x4Sneg,iL0N1Sneg+iL0N2Sneg+iL0N4Spos+iL0N5Sneg,1],
rxn[x5Spos,iL0N0Sneg+iL0N2Spos+iL0N3Sneg+iL0N5Sneg+iL0N7Sneg,1],
rxn[x5Sneg,iL0N0Spos+iL0N2Sneg+iL0N3Spos+iL0N5Spos+iL0N7Spos,1],
rxn[x6Spos,iL0N2Spos+iL0N4Spos+iL0N6Spos+iL0N7Sneg,1],
rxn[x6Sneg,iL0N2Sneg+iL0N4Sneg+iL0N6Sneg+iL0N7Spos,1],
rxn[x7Spos,iL0N0Spos+iL0N2Spos+iL0N4Spos+iL0N5Spos+iL0N6Spos,1],
rxn[x7Sneg,iL0N0Sneg+iL0N2Sneg+iL0N4Sneg+iL0N5Sneg+iL0N6Sneg,1],
rxn[x8Spos,iL0N0Sneg+iL0N3Sneg+iL0N4Spos+iL0N5Spos+iL0N6Spos+iL0N7Sneg,1],
rxn[x8Sneg,iL0N0Spos+iL0N3Spos+iL0N4Sneg+iL0N5Sneg+iL0N6Sneg+iL0N7Spos,1],
rxn[x9Spos,iL0N1Spos+iL0N2Spos+iL0N4Spos+iL0N6Spos,1],
rxn[x9Sneg,iL0N1Sneg+iL0N2Sneg+iL0N4Sneg+iL0N6Sneg,1],
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
rsys=VIRUS[0.652789,0,0,0.023067,0.461607,0,0,0.272034,0,0.438868,0,0.362966,0,0.623959,0,0.579211,0,0.149747,0,0.474706];
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
correctResults=List[-0.17813359690572 ,-2.456942485273304,-0.042613004025752,
 -1.217997660042281];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p0.svg"}], p0];

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
  WriteString["VIRUS_convergence_times.csv", "4.038955,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.880243,0,1.053501,0,1.529114,0,1.159462,0,1.181329,0,1.381449,0,1.932063,0,2.162209,0,0.964706,0,1.525284,0];
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
correctResults=List[-14.931822209730448, 10.436653742096384,-15.212276361472531,
  -9.05571633249446 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p1.svg"}], p1];

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
  WriteString["VIRUS_convergence_times.csv", "13.769360,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.061022,0,1.659602,0,0.588261,0,1.641145,0,1.047604,0,0,0.255142,0.301024,0,1.625134,0,0.265941,0,1.615191,0];
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
correctResults=List[-7.607292613267742, 7.071880230209787,-8.159969777248286,
 -9.378475892945511];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p2.svg"}], p2];

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
  WriteString["VIRUS_convergence_times.csv", "10.060066,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.197669,0,1.057072,0,0.964895,0,1.524228,0,1.636175,0,0.799803,0,0.445159,0,0.900400,0,0.944686,0,0.834337];
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
correctResults=List[ 7.587064932405931,-7.107709595017409,-4.035710899008529,
 -8.983196189353931];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p3.svg"}], p3];

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
  WriteString["VIRUS_convergence_times.csv", "10.304423,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.762404,0,1.292778,0,1.832227,0,0.462179,0,0.325927,0.747530,0,0,1.192358,0,0.978716,1.050472,0,0,1.316903];
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
correctResults=List[-1.487207401333099,-1.7619898790632  , 1.266460800401627,
 -1.137166252049451];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p4.svg"}], p4];

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
  WriteString["VIRUS_convergence_times.csv", "9.961491,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.867577,0,0.846525,0,1.041863,0,1.432318,0,1.212129,0,1.002422,0,0.411887,0,1.550701,0,1.033922,0,1.867359];
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
correctResults=List[ 6.75214303594826 ,-6.856721946053482,-3.541482416762129,
 -8.14827429289626 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p5.svg"}], p5];

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
  WriteString["VIRUS_convergence_times.csv", "11.266703,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.597877,0,1.133118,0,0.937324,0,1.216430,0,0.506771,0,1.176376,0,1.434652,0,0.494291,0,0.298472,0,0.465531];
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
correctResults=List[ 7.802647690594179,-7.446148493342376,-4.352479515446441,
 -9.198778947542179];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p6.svg"}], p6];

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
  WriteString["VIRUS_convergence_times.csv", "9.260840,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.289758,0,0.460996,0,0.933016,0,1.131520,0,0.364499,0,0.716085,0,0.562242,0,0.411636,0,1.713598,0,0.038011];
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
correctResults=List[ 6.970306943476182,-6.337656565957045,-4.630454001797454,
 -8.366438200424183];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p7.svg"}], p7];

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
  WriteString["VIRUS_convergence_times.csv", "6.621361,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.216459,0,0.146997,0,0,0.206410,0.251426,0,0,0.505615,0.178437,0,0.769104,0,0,0.623370,0,0.898334,0,0.228836];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p8.svg"}], p8];

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
  WriteString["VIRUS_convergence_times.csv", "5.024989,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.227034,0,0.704080,0,0.252074,0,0.796849,0,1.025974,0,1.745794,0,1.089244,0,0.411636,0,0.856443,0,0.762777];
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
correctResults=List[ 7.434399555742723,-7.401724793963408,-5.076304612530486,
 -8.830530812690723];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p9.svg"}], p9];

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
  WriteString["VIRUS_convergence_times.csv", "7.871903,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.553952,0,0.300414,0,0.515887,0,0,0.064729,0.650349,0,0,0.049916,0.182164,0,0,0.073652,0.053265,0,0.323456,0];
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
correctResults=List[-1.53422293368355 ,-1.761989879063199, 1.013125714864   ,
 -0.883831166511825];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p10.svg"}], p10];

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
  WriteString["VIRUS_convergence_times.csv", "2.767786,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.361556,0,0,0.255746,0.977986,0,0,0.109893,0,0.043231,0,0.115067,0,1.374966,0,1.078543,1.455253,0,0.139970,0];
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
correctResults=List[-1.378009804608925,-1.761989879063199, 1.157263203677452,
 -1.027968655325276];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p11.svg"}], p11];

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
  WriteString["VIRUS_convergence_times.csv", "5.912212,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.436315,0,0.786748,0,0.823595,0,0.806108,0,0.450186,0,0.908605,0,0.858379,0,1.164785,0,0.932044,0,1.968276];
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
correctResults=List[ 5.568018894016726,-6.386632957510924,-2.67030280627133 ,
 -6.964150150964725];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p12.svg"}], p12];

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
  WriteString["VIRUS_convergence_times.csv", "10.135040,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.523531,0,1.979867,0,0.445238,0,0.516994,0,0,0.148087,0,0.319967,0.756151,0,1.136748,0,0.068385,0,1.433541,0];
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
correctResults=List[-5.903615972717246, 7.884035611054856,-9.41854420298555 ,
 -9.744212228898386];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p13.svg"}], p13];

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
  WriteString["VIRUS_convergence_times.csv", "8.328507,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.959861,0,0.703346,0,0.342421,0,0,0.087310,0.812022,0,0.766099,0,0.513858,0,0,0.358041,0.378727,0,0,0.228836];
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
correctResults=List[-1.112634817499812,-1.7619898790632  , 0.412790987800365,
 -0.283496439448189];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p14.svg"}], p14];

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
  WriteString["VIRUS_convergence_times.csv", "5.150522,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.880243,0,0.695591,0,1.167248,0,2.156249,0,1.554564,0,0.973279,0,1.357822,0,2.162209,0,1.184325,0,2.061060,0];
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
correctResults=List[-15.313948779478373, 10.706694373390635,-12.783062241561337,
 -12.025011714994156];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p15.svg"}], p15];

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
  WriteString["VIRUS_convergence_times.csv", "14.192590,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.127345,0,0.370217,0,0.849323,0,0.468893,0,0.371810,0,1.103580,0,1.518587,0,0.405678,0,0,0.890649,0,0.476540];
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
correctResults=List[-2.806961031992133,-1.7619898790632  ,-1.281535226691956,
  1.410829775044132];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p16.svg"}], p16];

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
  WriteString["VIRUS_convergence_times.csv", "6.582622,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.746136,0,0.127890,0,0,0.408596,0,0.606480,0.511540,0,0.276490,0,0.233213,0,0.608919,0,0.253547,0,0.875746,0];
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
correctResults=List[-5.683233036771119,-1.209090007713125, 0.410186603849254,
 -1.386691798197227];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p17.svg"}], p17];

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
  WriteString["VIRUS_convergence_times.csv", "4.648557,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.590691,0,3.880838,0,2.029985,0,2.975990,0,1.817862,0,1.821217,0,3.315473,0,0.260557,0,0,0.670039,0,0.082048];
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
correctResults=List[ -9.04938570292755 , 11.171581232449968,-14.454610250832822,
 -11.283237423841339];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p18.svg"}], p18];

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
  WriteString["VIRUS_convergence_times.csv", "18.444700,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.471585,0,0.372086,0,0.330479,0,2.997277,0,1.400824,0.079082,0,0,0.471826,0,2.486336,0,0.763736,0,2.124788];
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
correctResults=List[ 6.171382475018484,-8.184150778703666,-2.61788458920838 ,
 -7.567513731966486];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p19.svg"}], p19];

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
  WriteString["VIRUS_convergence_times.csv", "12.498020,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.080889,0,0.666625,0,0.961449,0.051119,0,0,0.387362,0,1.327525,0,0.186103,0,0.411636,0,1.615688,0,0.217826];
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
correctResults=List[ 4.789231351882202,-4.591627993636107,-3.121727010084407,
 -6.185362608830203];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p20.svg"}], p20];

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
  WriteString["VIRUS_convergence_times.csv", "5.906222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.712759,0,2.055157,0,1.045764,0,1.146591,0,0.403913,0,0.231210,0,0.225086,0,0.948791,0,0,0.894119,1.615191,0];
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
correctResults=List[-4.694120362018048, 5.490917691490609,-6.493525258349702,
 -7.882995334405741];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p21.svg"}], p21];

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
  WriteString["VIRUS_convergence_times.csv", "10.278582,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.613514,0,0.231533,0.247646,0,0.563061,0,1.046218,0,0.414609,0,0.656593,0,0,0.191409,0.560667,0,0.145475,0];
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
correctResults=List[-2.602022948105033,-1.7619898790632  ,-1.076597142804856,
  1.205891691157031];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p22.svg"}], p22];

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
  WriteString["VIRUS_convergence_times.csv", "4.670725,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.331778,0,0.008335,0,0.390096,0,0,0.660000,0,0.227538,1.011392,0,0.447315,0,0,0.010811,0.112011,0,0,0.184799];
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
correctResults=List[-1.843865159329172,-1.761989879063199,-0.318439354028995,
  0.44773390238117 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p23.svg"}], p23];

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
  WriteString["VIRUS_convergence_times.csv", "3.384074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.787325,0,1.175492,0,0.341680,0.562159,0,0.357720,0,0.504844,0,0,0.328075,0,0.663378,0,0.147515,0,0.087551];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p24.svg"}], p24];

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
  WriteString["VIRUS_convergence_times.csv", "4.955738,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.197669,0,1.285400,0,0.992466,0,1.150264,0,0.318537,0,0.705660,0,0.525415,0,1.519376,0,0.401837,0,0.280212];
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
correctResults=List[ 4.472134928524476,-5.366812326960539,-1.703408418072478,
 -5.868266185472478];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p25.svg"}], p25];

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
  WriteString["VIRUS_convergence_times.csv", "8.376835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.289758,0,0.685352,0,0.677699,0,0.908407,0,3.192858,0,1.359450,0,0.762884,0.035613,0,0,0.195357,0.143639,0];
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
correctResults=List[ 4.77293956217168 ,-4.821992077879881,-3.306720556540505,
 -6.169070819119681];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p26.svg"}], p26];

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
  WriteString["VIRUS_convergence_times.csv", "8.251015,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.874330,0,0.070383,0,0,0.119677,0,0.832078,0.095579,0,0.439693,0,0.615957,0,0,0.490706,0,0.147019,0.692261,0];
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
correctResults=List[-2.797806623611604,-1.761989879063199, 0.741998967732653,
 -0.612704419380477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p27.svg"}], p27];

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
  WriteString["VIRUS_convergence_times.csv", "4.377682,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.721010,0,0.828175,0,0.933016,0,0.644192,0,0.925505,0,1.043142,0,0.473604,0,1.550701,0,0.217169,0,1.632497];
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
correctResults=List[ 1.938115666925889,-3.734050908141112,-0.456530688179748,
 -3.334246923873891];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p28.svg"}], p28];

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
  WriteString["VIRUS_convergence_times.csv", "8.969012,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.299678,0,0.532148,0,0.943522,0,1.728087,0,1.646257,0,1.563219,0,0.792722,0,2.086159,0,1.955717,0,0.567490,0];
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
correctResults=List[-11.251672795295558,  5.972166487476785, -9.114755005023861,
  -6.224263179703933];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p29.svg"}], p29];

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
  WriteString["VIRUS_convergence_times.csv", "12.115000,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.901986,0,2.149174,0,1.217220,0,1.011548,0,0.656816,0,0,0.676668,1.135337,0,1.061452,0,0,0.203039,1.250057,0];
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
correctResults=List[ -6.564318775747717,  8.367174515030344, -9.028922102498338,
 -11.100112137336572];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p30.svg"}], p30];

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
  WriteString["VIRUS_convergence_times.csv", "11.263297,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.745714,0,1.452460,0,1.017044,0,2.787653,0,1.047604,0,0,0.028091,0,0.457603,1.380942,0,1.722711,0,0.884920,0];
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
correctResults=List[ -6.018456218881994,  6.620845624706704, -6.021718170737953,
 -10.614658288449679];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p31.svg"}], p31];

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
  WriteString["VIRUS_convergence_times.csv", "11.524742,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.670741,0,0,0.479724,0,0.564256,0,0.997606,0,2.417518,0,1.649045,0.064826,0,0,0.020246,0,1.395325,0.325290,0];
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
correctResults=List[ 7.253905953213555,-6.09912450706241 ,-5.375050056574957,
 -8.650037210161555];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p32.svg"}], p32];

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
  WriteString["VIRUS_convergence_times.csv", "8.584578,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[2.509373,0,1.761186,0,0.541448,0,0.398436,0,0,0.492681,0.313300,0,1.211277,0,0,0.670359,0,2.273552,0,0.845345];
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
correctResults=List[-0.873086203115825,-1.701242531517333, 0.591592254638486,
 -0.583792401378043];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p33.svg"}], p33];

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
  WriteString["VIRUS_convergence_times.csv", "11.016958,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.184787,0,0.359601,1.442381,0,0,0.004885,0.050310,0,0.145862,0,0,0.076131,0.115814,0,0.655109,0,0.145475,0];
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
correctResults=List[-2.051086554665263,-1.7619898790632  ,-0.525660749365086,
  0.654955297717262];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p34.svg"}], p34];

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
  WriteString["VIRUS_convergence_times.csv", "4.180355,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.080889,0,0.405001,0,1.500515,0,0.283552,0,0.775148,0,0.522912,0,0.637673,0,0.337094,0,1.762678,0,0.217826];
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
correctResults=List[ 3.823237943768484,-3.596602604739641,-2.740309594048278,
 -5.219369200716485];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p35.svg"}], p35];

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
  WriteString["VIRUS_convergence_times.csv", "6.523288,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.260815,0,1.437493,0,1.350886,0,1.436157,0,1.165474,0,1.082232,0,1.086957,0,0.165554,0,0.737958,0,0.465531];
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
correctResults=List[  9.75058547717331 , -7.782343008570646, -5.768034813774841,
 -11.146716734121311];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p36.svg"}], p36];

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
  WriteString["VIRUS_convergence_times.csv", "10.189057,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.113404,0,1.209165,0,1.185461,0,0.226870,0,0.765679,0,0.360361,0,0.953366,0,1.035896,0,0.040186,0,1.203141];
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
correctResults=List[ 1.035835165849319,-2.96061748827873 ,-0.057954157565261,
 -2.431966422797319];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p37.svg"}], p37];

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
  WriteString["VIRUS_convergence_times.csv", "8.093527,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.164521,0,0.218102,0,0.479247,0,2.090821,0,2.485883,0,1.037930,0,0.035494,0,0.579400,0,1.468697,0,1.124244];
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
correctResults=List[  8.773502699404936, -7.969035794042325, -5.765536630375402,
 -10.169633956352937];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p38.svg"}], p38];

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
  WriteString["VIRUS_convergence_times.csv", "9.684338,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.046879,0,0.215643,0.589122,0,0.410179,0,0.559813,0,1.518916,0,0.185720,0,0.314528,0,1.245551,0,0,0.184799];
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
correctResults=List[-3.510962369281943,-1.7619898790632  ,-1.985536563981766,
  2.114831112333942];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p39.svg"}], p39];

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
  WriteString["VIRUS_convergence_times.csv", "6.271150,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.018377,0,0.498452,0,0.053622,0,0.886050,0,2.599978,0,1.134353,0,0.311313,0,0.225376,0,0.880983,0,0.038011];
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
correctResults=List[ 4.72971675296126 ,-5.412561996855115,-3.283605931175963,
 -6.125848009909261];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p40.svg"}], p40];

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
  WriteString["VIRUS_convergence_times.csv", "6.646514,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.349305,0,1.102851,0,1.005677,0,1.133327,0,1.123901,0,0.552555,0,0.391315,0,1.411809,0,0.489586,0,1.417820];
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
correctResults=List[ 3.98525951367615 ,-5.024001666359878,-1.329739300383345,
 -5.381390770624149];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p41.svg"}], p41];

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
  WriteString["VIRUS_convergence_times.csv", "9.978147,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.302007,0,0.432242,0,0.074013,0,0.683485,0.273650,0,1.001293,0,0.295182,0,0,0.114225,0.489279,0,0,0.096726];
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
correctResults=List[-2.824751297373469,-1.7619898790632  ,-1.299325492073292,
  1.428620040425468];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p42.svg"}], p42];

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
  WriteString["VIRUS_convergence_times.csv", "3.762102,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.228723,0,0.999942,0,1.406028,0,0.556798,0,0.718562,0,0.642790,0,0.471826,0,0.475043,0,0.169330,0,1.755433];
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
correctResults=List[ 1.286810097109068,-2.608216414256676,-0.661330162847065,
 -4.103587692838191];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p43.svg"}], p43];

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
  WriteString["VIRUS_convergence_times.csv", "7.424476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.394086,0,0.778802,0,0.961449,0,0.283552,0,0.957609,0,0.490662,0,0.762884,0,0.914931,0,0.831903,0,0.038011];
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
correctResults=List[ 1.542337875187379,-3.021828838639235,-0.735435960663573,
 -2.93846913213538 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p44.svg"}], p44];

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
  WriteString["VIRUS_convergence_times.csv", "6.413888,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.106860,0,0,0.049740,0,0.961449,0,1.399122,0,3.785969,0,1.456198,0.190290,0,0,0.765847,0,1.248335,0,0.581127];
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
correctResults=List[ 6.271790596961957,-5.833722365103935,-4.197359355015771,
 -7.667921853909959];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p45.svg"}], p45];

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
  WriteString["VIRUS_convergence_times.csv", "10.544937,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.331573,0,0.685352,0,0.848006,0,0.372752,0,0.364499,0,0.812508,0,0.461921,0,0.299918,0,1.199256,0,1.305894];
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
correctResults=List[ 3.942167680084688,-4.235141553931212,-2.512831358088271,
 -5.338298937032689];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p46.svg"}], p46];

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
  WriteString["VIRUS_convergence_times.csv", "6.681678,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.367678,0,0.697081,0,1.166219,0,0.864145,0,0.228692,0,0.431049,0,0.387252,0,0.930593,0,0.315823,0,1.544425];
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
correctResults=List[ 2.319204386642566,-3.686409229441269,-0.615531637195969,
 -3.715335643590567];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p47.svg"}], p47];

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
  WriteString["VIRUS_convergence_times.csv", "7.932958,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.005479,0,0.695591,0,0.833239,0,0.627874,0,1.510681,0,1.570060,0,2.062605,0,1.003140,0,0.110770,0,1.525284,0];
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
correctResults=List[-12.883041478310528,  5.345440530679185, -9.846285021948145,
  -4.239281249184451];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p48.svg"}], p48];

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
  WriteString["VIRUS_convergence_times.csv", "10.944724,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.886578,0,0.761800,0,1.359956,0,0.392565,0,1.228446,0,2.276622,0,1.441887,0,0.623450,0,0.946362,0,1.163818,0];
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
correctResults=List[-11.653943746899069,  4.234419712326486, -8.218711874532028,
  -3.644812759895167];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p49.svg"}], p49];

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
  WriteString["VIRUS_convergence_times.csv", "11.081484,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.044347,0,0.606113,0,1.334109,0,1.735314,0,1.027740,0,1.287305,0,1.201117,0,1.256580,0,1.526890,0,0.631711,0];
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
correctResults=List[-7.079084894378622, 4.663599112532098,-9.005040839257026,
 -3.716842595581393];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p50.svg"}], p50];

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
  WriteString["VIRUS_convergence_times.csv", "10.651226,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.420479,0,0.534607,0,1.167248,0,1.225855,0,1.335152,0,2.072373,0,1.592750,0,1.111838,0,1.526890,0,1.703265,0];
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
correctResults=List[-12.81508853993267 ,  7.092869529030282,-12.409876904534624,
  -5.170547363300166];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p51.svg"}], p51];

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
  WriteString["VIRUS_convergence_times.csv", "12.690456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.446456,0,0.276767,0,0.943522,0,1.400191,0,1.795228,0,1.074914,0,1.290517,0,1.861024,0,1.551678,0,1.361982,0];
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
correctResults=List[-14.698608576431575,  6.604321369430978, -8.672352246529979,
  -7.930975702106201];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p52.svg"}], p52];

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
  WriteString["VIRUS_convergence_times.csv", "12.002278,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.065255,0,0,0.834986,0,0.989881,0,0.952894,0,0.775148,0,0.651585,0,0.813170,0,0.803024,0,1.150424,0,0.217826];
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
correctResults=List[ 4.803912732124311,-4.938096017591929,-3.251383779419676,
 -6.200043989072312];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p53.svg"}], p53];

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
  WriteString["VIRUS_convergence_times.csv", "7.254192,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.998088,0,1.226591,0,1.360244,0,1.101652,0,0.541797,0,0,0.709243,0.275626,0,1.700242,0,0.216615,0,0.884920,0];
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
correctResults=List[-5.075401066913904, 4.586529651186425,-4.904600895531101,
 -7.663143616615973];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p54.svg"}], p54];

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
  WriteString["VIRUS_convergence_times.csv", "9.015018,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.586678,0,2.544729,0,0.673845,0,1.978301,0,1.001642,0,0,0.903718,1.084797,0,1.174302,0,0,0.128925,1.068406,0];
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
correctResults=List[ -6.482582770405799,  9.059142479202707, -9.596732171496482,
 -11.916237996683154];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p55.svg"}], p55];

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
  WriteString["VIRUS_convergence_times.csv", "12.145343,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.065255,0,0,0.909708,0,0.621121,0,0.774493,0,0.752282,0,1.005680,0,0.788028,0,1.026649,0,1.346245,0,0.038011];
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
correctResults=List[ 5.819136652469617,-6.099466831438517,-3.820369539631621,
 -7.215267909417618];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p56.svg"}], p56];

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
  WriteString["VIRUS_convergence_times.csv", "7.327472,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.190702,0,0.570360,0,1.445540,0,1.314604,0,1.203272,0,1.350176,0,1.174958,0,1.600791,0,1.526890,0,0.453730,0];
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
correctResults=List[-8.502669095173182, 4.568703302643258,-9.042887799031659,
 -3.489204016029082];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p57.svg"}], p57];

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
  WriteString["VIRUS_convergence_times.csv", "10.831023,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.138107,0,1.068067,0,1.263746,0,0.931607,0,0.333700,0,1.487970,0,2.610433,0,1.478505,0,0.466474,0,1.163818,0];
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
correctResults=List[-11.720351934050024,  9.871225812933405,-14.98988633518152 ,
  -8.147250500459512];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p58.svg"}], p58];

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
  WriteString["VIRUS_convergence_times.csv", "11.942428,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.080889,0,0.629358,0,0.848006,0,0.752136,0,0.843513,0,0.201066,0,0.788028,0,0.318411,0,0.880983,0,0.399476];
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
correctResults=List[ 2.236120540022832,-3.153911368005251,-1.663791654480712,
 -3.632251796970833];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p59.svg"}], p59];

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
  WriteString["VIRUS_convergence_times.csv", "5.741866,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.540231,0,0.517180,0,0.422382,0,0.618223,0.091651,0,0,1.584871,0,0.512209,0,0.579400,0,1.762678,0.143639,0];
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
correctResults=List[ 8.329057733833771,-7.914052643828368,-5.277978775872008,
 -9.725188990781772];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p60.svg"}], p60];

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
  WriteString["VIRUS_convergence_times.csv", "6.772464,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.810338,0,0.051088,0,0,0.148397,0.567353,0,0.858445,0,0,0.670153,0,0.558178,0,0.528636,0.929507,0,0.323456,0];
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
correctResults=List[-2.963423281697279,-1.761989879063199, 2.742676680765806,
 -2.613382132413631];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p61.svg"}], p61];

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
  WriteString["VIRUS_convergence_times.csv", "5.445549,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.307489,0,1.084501,0,0.678848,0,0.671291,0,0.815337,0,0.797849,0,0.391315,0,1.220833,0,0.824714,0,1.456352];
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
correctResults=List[ 4.452878485739214,-5.527436860852217,-1.739138511790053,
 -5.849009742687214];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p62.svg"}], p62];

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
  WriteString["VIRUS_convergence_times.csv", "9.248528,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.263977,0,1.374333,0,1.039446,0,0.509768,0,0.810868,0,1.375259,0,1.788057,0,0.556080,0,0.466474,0,0.965653,0];
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
correctResults=List[-9.223125278566778, 5.950558671257455,-9.490534923123644,
 -5.805267629165491];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p63.svg"}], p63];

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
  WriteString["VIRUS_convergence_times.csv", "10.149916,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.244554,0,1.231108,0,1.259845,0,0.209255,0,0.197974,0,0.347982,0,1.336616,0,0.439565,0.117711,0,0,0.929750];
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
correctResults=List[ 1.178031896065463,-2.925820680363758,-0.234947695696376,
 -3.229711739581107];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p64.svg"}], p64];

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
  WriteString["VIRUS_convergence_times.csv", "7.314360,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.253212,0,0.983485,0,0.588094,1.468390,0,0.123755,0,1.687657,0,0,0.344075,0,0.344831,1.559112,0,0,0.755438];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p65.svg"}], p65];

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
  WriteString["VIRUS_convergence_times.csv", "9.108050,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.298416,0,0.055604,0.276079,0,0,0.616416,0.351484,0,1.429334,0,0.656593,0,0,0.119131,0.371786,0,0,0.019662];
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
correctResults=List[-3.693534644682105,-1.7619898790632  ,-2.168108839381928,
  2.297403387734104];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p66.svg"}], p66];

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
  WriteString["VIRUS_convergence_times.csv", "4.194506,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.276024,0,0.031982,0,0.515887,0,0,0.899825,0.026059,0,0.766099,0,0,0.405030,0,1.040613,0.654116,0,1.244551,0];
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
correctResults=List[-3.328320718590176,-1.761989879063199, 1.463841089065537,
 -1.334546540713361];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p67.svg"}], p67];

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
  WriteString["VIRUS_convergence_times.csv", "5.860186,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.644559,0,0.760074,0,0.734563,0,1.287565,0,0.912109,0,0.587412,0,1.339918,0,1.287637,0,1.125884,0.143639,0];
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
correctResults=List[ 6.14391336899994 ,-6.846981086783384,-3.710523186100737,
 -7.540044625947941];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p68.svg"}], p68];

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
  WriteString["VIRUS_convergence_times.csv", "8.823362,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.579714,0,1.304317,0,0.642661,0.415825,0,0.243163,0,0,0.675365,0,1.953017,0,0.352757,0.243385,0,0,0.656356];
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
correctResults=List[ 0.331777673025708,-3.061970421128248, 0.596305357006178,
 -2.587190260400644];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p69.svg"}], p69];

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
  WriteString["VIRUS_convergence_times.csv", "8.066561,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.232307,0,0,0.498452,0,0.479247,0,0.283552,0.319612,0,0.152703,0,0,0.487065,0.035613,0,0,1.321707,0.143639,0];
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
correctResults=List[ 0.601942725397775,-2.047003330290737,-0.822689326329247,
 -1.998073982345776];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p70.svg"}], p70];

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
  WriteString["VIRUS_convergence_times.csv", "3.953896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.572114,0,0.940566,0,0.655177,0,1.493908,0,1.407442,0,1.901027,0,1.052542,0,1.298663,0,1.628520,0,1.760145,0];
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
correctResults=List[-12.120478003397288,  8.014076837799509,-12.757924221045895,
  -6.664914664327346];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p71.svg"}], p71];

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
  WriteString["VIRUS_convergence_times.csv", "12.710105,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.387319,0,0.619332,0,0.992466,0,0.402786,0,0.718562,0,0.956817,0,1.033622,0,0.958523,0,0.453642,0,1.386627];
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
correctResults=List[ 3.065162102997286,-4.509570219569182,-1.217369494808928,
 -4.461293359945286];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p72.svg"}], p72];

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
  WriteString["VIRUS_convergence_times.csv", "8.909697,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.023439,0,0,0.647897,0.400147,0,0,0.171994,0.068786,0,0,1.295276,0,0.210993,0.240742,0,0,1.493237,0,1.124244];
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
correctResults=List[ 5.529951918244344,-5.991882503979181,-3.917091471923606,
 -6.926083175192345];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p73.svg"}], p73];

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
  WriteString["VIRUS_convergence_times.csv", "5.676754,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.227034,0,0.797530,0,1.018026,0,0.774493,0,2.189329,0,1.295276,0,0.361601,0,0.243870,0,1.027972,0,0.217826];
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
correctResults=List[ 6.157389263927919,-5.398891218953108,-4.007188884390609,
 -7.55352052087592 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p74.svg"}], p74];

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
  WriteString["VIRUS_convergence_times.csv", "8.152955,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.886578,0,1.119143,0,1.135657,0,0.111414,0,0.781073,0,1.826104,0,1.788057,0,1.208647,0,1.148135,0,1.163818,0];
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
correctResults=List[-12.711192309473457,  7.381608927152116,-12.428392327474878,
  -5.729510736603579];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p75.svg"}], p75];

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
  WriteString["VIRUS_convergence_times.csv", "11.168626,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.059981,0,1.152603,0,1.103323,0,0.752136,0,0.569823,0,1.906717,0,0.963779,0,0.821706,0,1.468697,0,0.217826];
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
correctResults=List[  9.318595360775571, -8.163411074333167, -5.950565649482622,
 -11.706039806117884];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p76.svg"}], p76];

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
  WriteString["VIRUS_convergence_times.csv", "9.016592,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.796610,0,0.695591,0,1.000100,0,1.137332,0,1.708156,0,0.973279,0,1.514271,0,2.342618,0,1.184325,0,1.525284,0];
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
correctResults=List[-16.892086773290934,  8.969083675644358,-11.299207113273068,
 -10.03364544778987 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p77.svg"}], p77];

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
  WriteString["VIRUS_convergence_times.csv", "12.877565,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.416683,0,0.359601,0.304798,0,0,0.310651,0,0.111825,0.653387,0,0.133400,0,0.097699,0,0.678904,0,0,0.349936];
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
correctResults=List[-1.630143316824133,-1.7619898790632  ,-0.104717511523957,
  0.234012059876132];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p78.svg"}], p78];

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
  WriteString["VIRUS_convergence_times.csv", "3.416885,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.310665,0,0.479724,0,0.479247,0,1.711662,0,0.638420,0,0.877007,0,1.139277,0,0.448999,0,0.611541,0,0.217826];
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
correctResults=List[ 6.485365282118303,-6.68611220365045 ,-4.466869292629974,
 -7.881496539066304];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p79.svg"}], p79];

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
  WriteString["VIRUS_convergence_times.csv", "6.914368,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.237789,0,0.319710,0,0,0.090671,0,0.087310,0,0.574905,0,0.147969,0,0.430682,0,0.452775,0,0.447447,0.139970,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p80.svg"}], p80];

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
  WriteString["VIRUS_convergence_times.csv", "3.929226,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.802735,0,0.685186,0,1.135657,0,1.728087,0,1.735640,0,1.600681,0,1.701452,0,1.433592,0,1.072532,0,1.561982,0];
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
correctResults=List[-13.349475412741008,  7.929913648865182,-11.74714615154974 ,
  -7.507366355954849];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p81.svg"}], p81];

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
  WriteString["VIRUS_convergence_times.csv", "13.457542,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.259120,0,0.242906,0,0,0.292857,0.251426,0,0,0.389904,0,0.213119,0.615957,0,0,0.263308,0,0.547836,0.508776,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p82.svg"}], p82];

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
  WriteString["VIRUS_convergence_times.csv", "4.585208,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.504111,0,0.480882,0,0.805381,0,1.668922,0,1.510681,0,1.130292,0,1.044413,0,1.637024,0,1.769807,0,1.345469,0];
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
correctResults=List[-12.441218345537486,  7.144164571068246, -9.378529570586606,
  -8.30448478132411 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p83.svg"}], p83];

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
  WriteString["VIRUS_convergence_times.csv", "11.896982,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.253222,0.276578,0,1.006706,0,0,0.165897,0.690768,0,1.405228,0,0.309153,0,0.206775,0,0.852913,0,0,0.806814];
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
correctResults=List[-2.704313265521224,-1.7619898790632  ,-1.178887460221047,
  1.308182008573223];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p84.svg"}], p84];

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
  WriteString["VIRUS_convergence_times.csv", "5.974054,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.655957,0,0.838414,0,1.231867,0,1.048583,0,1.258472,0,1.713392,0,1.485063,0,0.871042,0,0.845230,0,0.765656,0];
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
correctResults=List[-8.765936642502249, 4.135345587036569,-8.51312150721483 ,
 -3.152254876632532];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p85.svg"}], p85];

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
  WriteString["VIRUS_convergence_times.csv", "10.713676,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.461873,0,0,0.012473,0,0.337372,0,1.354634,0,3.261454,0,0.522912,0.165147,0,0.557590,0,0,1.297167,0,0.038011];
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
correctResults=List[ 4.404000579684955,-4.232780928738689,-3.650790229288952,
 -5.800131836632956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p86.svg"}], p86];

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
  WriteString["VIRUS_convergence_times.csv", "8.008633,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.821960,0,0.411055,0.116110,0,0,0.478664,0.102046,0,0.421450,0,0,0.512716,0,0.617709,0.778303,0,0,0.096726];
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
correctResults=List[-1.07705316915005 ,-1.7619898790632  , 0.448372636150127,
 -0.319078087797951];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p87.svg"}], p87];

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
  WriteString["VIRUS_convergence_times.csv", "4.356737,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.410543,0,1.399871,0,1.039446,0,0.790918,0,0.840661,0,1.938814,0,1.961269,0,1.951041,0,0.744345,0,1.163818,0];
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
correctResults=List[-13.439245193813742, 10.634543368122538,-16.038652203130052,
  -8.625119742889245];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p88.svg"}], p88];

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
  WriteString["VIRUS_convergence_times.csv", "13.240726,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.513611,0,1.190247,0,1.323316,0,1.436157,0,0.412770,0,1.521675,0,1.220803,0,0.842464,0,0.427864,0,0.834337];
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
correctResults=List[  9.262123238106348, -8.208008029513335, -5.144673077675714,
 -10.841755082120764];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p89.svg"}], p89];

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
  WriteString["VIRUS_convergence_times.csv", "10.723244,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.485945,0,1.175492,0,1.761289,0.675974,0,0.032295,0,0,0.466230,0,1.164421,0,1.245553,1.849129,0,0,2.128092];
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
correctResults=List[-2.679244629827981,-1.7619898790632  , 2.458498028896509,
 -2.329203480544333];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p90.svg"}], p90];

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
  WriteString["VIRUS_convergence_times.csv", "11.984420,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.489749,0,0.223800,0,0.515887,0,0.318946,0,0.581061,0,0.439693,0,0,0.762375,0,0.073652,0.278584,0,0.323456,0];
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
correctResults=List[-0.943496582182749,-1.761989879063199, 0.722749981251276,
 -0.5934554328991  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p91.svg"}], p91];

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
  WriteString["VIRUS_convergence_times.csv", "4.007204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[1.187315,0,2.393961,0,0.931172,0,1.596207,0,0.495835,0,0.782712,0,0.124002,0,0.892365,0,0.488040,0,1.250057,0];
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
correctResults=List[-5.284721425493201, 7.194359668991525,-8.672456860484864,
 -9.11094768727241 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p92.svg"}], p92];

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
  WriteString["VIRUS_convergence_times.csv", "10.141665,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.429346,0,0.657544,0,1.433599,0,0.952442,0,0.883239,0,0.736932,0,0.391569,0,1.383880,0,0.892880,0,0.834337];
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
correctResults=List[ 4.423230002939683,-5.044641443543409,-1.726188985480086,
 -5.819361259887684];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p93.svg"}], p93];

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
  WriteString["VIRUS_convergence_times.csv", "9.595769,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,1.336210,0,0.823257,0,1.690352,0.078444,0,0,0.195666,0,0.132332,0,0.021525,0,0.905873,0.997177,0,0,0.087551];
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
correctResults=List[-1.331677142319923,-1.7619898790632  , 1.110930541388451,
 -0.981635993036275];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p94.svg"}], p94];

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
  WriteString["VIRUS_convergence_times.csv", "6.268389,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0.253004,0,0,0.909708,0,0.621121,0,0.997606,0,0.387362,0,1.230777,0,0.411887,0,0.150647,0,1.591148,0,0.581127];
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
correctResults=List[ 8.071917574663738,-6.921805360369658,-5.567785172078028,
 -9.745394398679604];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p95.svg"}], p95];

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
  WriteString["VIRUS_convergence_times.csv", "7.134387,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.695879,0,1.074854,0.021910,0,0,0.346782,0,0.525709,0,1.237291,0,1.598467,0,0.621671,1.673137,0,0,1.254519];
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
correctResults=List[-1.890466854710235,-2.117942344025197, 1.669720253778763,
 -2.252330635350582];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p96.svg"}], p96];

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
  WriteString["VIRUS_convergence_times.csv", "9.050219,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.310665,0,0.386274,0,1.018026,0,1.265208,0,2.850803,0,0.973431,0,0.637673,0.166201,0,0,0.538170,0,0.399476];
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
correctResults=List[ 5.141855861365777,-4.705474981121992,-3.640211475862281,
 -6.537987118313779];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p97.svg"}], p97];

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
  WriteString["VIRUS_convergence_times.csv", "8.545927,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.215418,0.185399,0,0.544607,0,1.696021,0,0.742734,0,0.341640,0,0.054414,0,0.286600,0,0.579009,0,0,0.412320];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p98.svg"}], p98];

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
  WriteString["VIRUS_convergence_times.csv", "5.058163,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUS_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUS_convergence_times.csv", ""];
];

rsys=VIRUS[0,0.059981,0,0.610630,0,1.046459,0,1.845350,0,2.326289,0,1.263026,0,0.336457,0.277919,0,0,0.807611,0.143639,0];
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
correctResults=List[ 8.567917640626412,-6.627852791004156,-5.953424272907989,
 -9.964048897574413];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUS_p99.svg"}], p99];

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
  WriteString["VIRUS_convergence_times.csv", "8.717360,"];
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
rsys=VIRUSReduced[0.652789,0,0,0.023067,0.461607,0,0,0.272034,0,0.438868,0,0.362966,0,0.623959,0,0.579211,0,0.149747,0,0.474706];
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
correctResults=List[-0.17813359690572 ,-2.456942485273304,-0.042613004025752,
 -1.217997660042281];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p0.svg"}], p0];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.038955,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.880243,0,1.053501,0,1.529114,0,1.159462,0,1.181329,0,1.381449,0,1.932063,0,2.162209,0,0.964706,0,1.525284,0];
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
correctResults=List[-14.931822209730448, 10.436653742096384,-15.212276361472531,
  -9.05571633249446 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p1.svg"}], p1];

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
  WriteString["VIRUSReduced_convergence_times.csv", "13.769360,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.061022,0,1.659602,0,0.588261,0,1.641145,0,1.047604,0,0,0.255142,0.301024,0,1.625134,0,0.265941,0,1.615191,0];
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
correctResults=List[-7.607292613267742, 7.071880230209787,-8.159969777248286,
 -9.378475892945511];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p2.svg"}], p2];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.060066,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.197669,0,1.057072,0,0.964895,0,1.524228,0,1.636175,0,0.799803,0,0.445159,0,0.900400,0,0.944686,0,0.834337];
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
correctResults=List[ 7.587064932405931,-7.107709595017409,-4.035710899008529,
 -8.983196189353931];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p3.svg"}], p3];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.304423,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.762404,0,1.292778,0,1.832227,0,0.462179,0,0.325927,0.747530,0,0,1.192358,0,0.978716,1.050472,0,0,1.316903];
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
correctResults=List[-1.487207401333099,-1.7619898790632  , 1.266460800401627,
 -1.137166252049451];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p4.svg"}], p4];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.961491,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.867577,0,0.846525,0,1.041863,0,1.432318,0,1.212129,0,1.002422,0,0.411887,0,1.550701,0,1.033922,0,1.867359];
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
correctResults=List[ 6.75214303594826 ,-6.856721946053482,-3.541482416762129,
 -8.14827429289626 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p5.svg"}], p5];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.266703,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.597877,0,1.133118,0,0.937324,0,1.216430,0,0.506771,0,1.176376,0,1.434652,0,0.494291,0,0.298472,0,0.465531];
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
correctResults=List[ 7.802647690594179,-7.446148493342376,-4.352479515446441,
 -9.198778947542179];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p6.svg"}], p6];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.260840,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.289758,0,0.460996,0,0.933016,0,1.131520,0,0.364499,0,0.716085,0,0.562242,0,0.411636,0,1.713598,0,0.038011];
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
correctResults=List[ 6.970306943476182,-6.337656565957045,-4.630454001797454,
 -8.366438200424183];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p7.svg"}], p7];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.621361,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.216459,0,0.146997,0,0,0.206410,0.251426,0,0,0.505615,0.178437,0,0.769104,0,0,0.623370,0,0.898334,0,0.228836];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p8.svg"}], p8];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.024989,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.227034,0,0.704080,0,0.252074,0,0.796849,0,1.025974,0,1.745794,0,1.089244,0,0.411636,0,0.856443,0,0.762777];
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
correctResults=List[ 7.434399555742723,-7.401724793963408,-5.076304612530486,
 -8.830530812690723];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p9.svg"}], p9];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.871903,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.553952,0,0.300414,0,0.515887,0,0,0.064729,0.650349,0,0,0.049916,0.182164,0,0,0.073652,0.053265,0,0.323456,0];
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
correctResults=List[-1.53422293368355 ,-1.761989879063199, 1.013125714864   ,
 -0.883831166511825];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p10.svg"}], p10];

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
  WriteString["VIRUSReduced_convergence_times.csv", "2.767786,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.361556,0,0,0.255746,0.977986,0,0,0.109893,0,0.043231,0,0.115067,0,1.374966,0,1.078543,1.455253,0,0.139970,0];
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
correctResults=List[-1.378009804608925,-1.761989879063199, 1.157263203677452,
 -1.027968655325276];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p11.svg"}], p11];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.912212,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.436315,0,0.786748,0,0.823595,0,0.806108,0,0.450186,0,0.908605,0,0.858379,0,1.164785,0,0.932044,0,1.968276];
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
correctResults=List[ 5.568018894016726,-6.386632957510924,-2.67030280627133 ,
 -6.964150150964725];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p12.svg"}], p12];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.135040,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.523531,0,1.979867,0,0.445238,0,0.516994,0,0,0.148087,0,0.319967,0.756151,0,1.136748,0,0.068385,0,1.433541,0];
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
correctResults=List[-5.903615972717246, 7.884035611054856,-9.41854420298555 ,
 -9.744212228898386];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p13.svg"}], p13];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.328507,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.959861,0,0.703346,0,0.342421,0,0,0.087310,0.812022,0,0.766099,0,0.513858,0,0,0.358041,0.378727,0,0,0.228836];
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
correctResults=List[-1.112634817499812,-1.7619898790632  , 0.412790987800365,
 -0.283496439448189];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p14.svg"}], p14];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.150522,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.880243,0,0.695591,0,1.167248,0,2.156249,0,1.554564,0,0.973279,0,1.357822,0,2.162209,0,1.184325,0,2.061060,0];
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
correctResults=List[-15.313948779478373, 10.706694373390635,-12.783062241561337,
 -12.025011714994156];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p15.svg"}], p15];

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
  WriteString["VIRUSReduced_convergence_times.csv", "14.192590,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.127345,0,0.370217,0,0.849323,0,0.468893,0,0.371810,0,1.103580,0,1.518587,0,0.405678,0,0,0.890649,0,0.476540];
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
correctResults=List[-2.806961031992133,-1.7619898790632  ,-1.281535226691956,
  1.410829775044132];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p16.svg"}], p16];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.582622,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.746136,0,0.127890,0,0,0.408596,0,0.606480,0.511540,0,0.276490,0,0.233213,0,0.608919,0,0.253547,0,0.875746,0];
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
correctResults=List[-5.683233036771119,-1.209090007713125, 0.410186603849254,
 -1.386691798197227];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p17.svg"}], p17];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.648557,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.590691,0,3.880838,0,2.029985,0,2.975990,0,1.817862,0,1.821217,0,3.315473,0,0.260557,0,0,0.670039,0,0.082048];
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
correctResults=List[ -9.04938570292755 , 11.171581232449968,-14.454610250832822,
 -11.283237423841339];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p18.svg"}], p18];

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
  WriteString["VIRUSReduced_convergence_times.csv", "18.444700,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.471585,0,0.372086,0,0.330479,0,2.997277,0,1.400824,0.079082,0,0,0.471826,0,2.486336,0,0.763736,0,2.124788];
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
correctResults=List[ 6.171382475018484,-8.184150778703666,-2.61788458920838 ,
 -7.567513731966486];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p19.svg"}], p19];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.498020,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.080889,0,0.666625,0,0.961449,0.051119,0,0,0.387362,0,1.327525,0,0.186103,0,0.411636,0,1.615688,0,0.217826];
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
correctResults=List[ 4.789231351882202,-4.591627993636107,-3.121727010084407,
 -6.185362608830203];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p20.svg"}], p20];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.906222,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.712759,0,2.055157,0,1.045764,0,1.146591,0,0.403913,0,0.231210,0,0.225086,0,0.948791,0,0,0.894119,1.615191,0];
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
correctResults=List[-4.694120362018048, 5.490917691490609,-6.493525258349702,
 -7.882995334405741];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p21.svg"}], p21];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.278582,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.613514,0,0.231533,0.247646,0,0.563061,0,1.046218,0,0.414609,0,0.656593,0,0,0.191409,0.560667,0,0.145475,0];
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
correctResults=List[-2.602022948105033,-1.7619898790632  ,-1.076597142804856,
  1.205891691157031];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p22.svg"}], p22];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.670725,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.331778,0,0.008335,0,0.390096,0,0,0.660000,0,0.227538,1.011392,0,0.447315,0,0,0.010811,0.112011,0,0,0.184799];
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
correctResults=List[-1.843865159329172,-1.761989879063199,-0.318439354028995,
  0.44773390238117 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p23.svg"}], p23];

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
  WriteString["VIRUSReduced_convergence_times.csv", "3.384074,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.787325,0,1.175492,0,0.341680,0.562159,0,0.357720,0,0.504844,0,0,0.328075,0,0.663378,0,0.147515,0,0.087551];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p24.svg"}], p24];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.955738,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.197669,0,1.285400,0,0.992466,0,1.150264,0,0.318537,0,0.705660,0,0.525415,0,1.519376,0,0.401837,0,0.280212];
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
correctResults=List[ 4.472134928524476,-5.366812326960539,-1.703408418072478,
 -5.868266185472478];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p25.svg"}], p25];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.376835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.289758,0,0.685352,0,0.677699,0,0.908407,0,3.192858,0,1.359450,0,0.762884,0.035613,0,0,0.195357,0.143639,0];
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
correctResults=List[ 4.77293956217168 ,-4.821992077879881,-3.306720556540505,
 -6.169070819119681];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p26.svg"}], p26];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.251015,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.874330,0,0.070383,0,0,0.119677,0,0.832078,0.095579,0,0.439693,0,0.615957,0,0,0.490706,0,0.147019,0.692261,0];
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
correctResults=List[-2.797806623611604,-1.761989879063199, 0.741998967732653,
 -0.612704419380477];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p27.svg"}], p27];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.377682,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.721010,0,0.828175,0,0.933016,0,0.644192,0,0.925505,0,1.043142,0,0.473604,0,1.550701,0,0.217169,0,1.632497];
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
correctResults=List[ 1.938115666925889,-3.734050908141112,-0.456530688179748,
 -3.334246923873891];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p28.svg"}], p28];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.969012,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.299678,0,0.532148,0,0.943522,0,1.728087,0,1.646257,0,1.563219,0,0.792722,0,2.086159,0,1.955717,0,0.567490,0];
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
correctResults=List[-11.251672795295558,  5.972166487476785, -9.114755005023861,
  -6.224263179703933];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p29.svg"}], p29];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.115000,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.901986,0,2.149174,0,1.217220,0,1.011548,0,0.656816,0,0,0.676668,1.135337,0,1.061452,0,0,0.203039,1.250057,0];
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
correctResults=List[ -6.564318775747717,  8.367174515030344, -9.028922102498338,
 -11.100112137336572];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p30.svg"}], p30];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.263297,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.745714,0,1.452460,0,1.017044,0,2.787653,0,1.047604,0,0,0.028091,0,0.457603,1.380942,0,1.722711,0,0.884920,0];
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
correctResults=List[ -6.018456218881994,  6.620845624706704, -6.021718170737953,
 -10.614658288449679];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p31.svg"}], p31];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.524742,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.670741,0,0,0.479724,0,0.564256,0,0.997606,0,2.417518,0,1.649045,0.064826,0,0,0.020246,0,1.395325,0.325290,0];
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
correctResults=List[ 7.253905953213555,-6.09912450706241 ,-5.375050056574957,
 -8.650037210161555];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p32.svg"}], p32];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.584578,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[2.509373,0,1.761186,0,0.541448,0,0.398436,0,0,0.492681,0.313300,0,1.211277,0,0,0.670359,0,2.273552,0,0.845345];
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
correctResults=List[-0.873086203115825,-1.701242531517333, 0.591592254638486,
 -0.583792401378043];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p33.svg"}], p33];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.016958,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.184787,0,0.359601,1.442381,0,0,0.004885,0.050310,0,0.145862,0,0,0.076131,0.115814,0,0.655109,0,0.145475,0];
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
correctResults=List[-2.051086554665263,-1.7619898790632  ,-0.525660749365086,
  0.654955297717262];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p34.svg"}], p34];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.180355,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.080889,0,0.405001,0,1.500515,0,0.283552,0,0.775148,0,0.522912,0,0.637673,0,0.337094,0,1.762678,0,0.217826];
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
correctResults=List[ 3.823237943768484,-3.596602604739641,-2.740309594048278,
 -5.219369200716485];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p35.svg"}], p35];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.523288,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.260815,0,1.437493,0,1.350886,0,1.436157,0,1.165474,0,1.082232,0,1.086957,0,0.165554,0,0.737958,0,0.465531];
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
correctResults=List[  9.75058547717331 , -7.782343008570646, -5.768034813774841,
 -11.146716734121311];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p36.svg"}], p36];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.189057,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.113404,0,1.209165,0,1.185461,0,0.226870,0,0.765679,0,0.360361,0,0.953366,0,1.035896,0,0.040186,0,1.203141];
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
correctResults=List[ 1.035835165849319,-2.96061748827873 ,-0.057954157565261,
 -2.431966422797319];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p37.svg"}], p37];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.093527,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.164521,0,0.218102,0,0.479247,0,2.090821,0,2.485883,0,1.037930,0,0.035494,0,0.579400,0,1.468697,0,1.124244];
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
correctResults=List[  8.773502699404936, -7.969035794042325, -5.765536630375402,
 -10.169633956352937];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p38.svg"}], p38];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.684338,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.046879,0,0.215643,0.589122,0,0.410179,0,0.559813,0,1.518916,0,0.185720,0,0.314528,0,1.245551,0,0,0.184799];
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
correctResults=List[-3.510962369281943,-1.7619898790632  ,-1.985536563981766,
  2.114831112333942];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p39.svg"}], p39];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.271150,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.018377,0,0.498452,0,0.053622,0,0.886050,0,2.599978,0,1.134353,0,0.311313,0,0.225376,0,0.880983,0,0.038011];
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
correctResults=List[ 4.72971675296126 ,-5.412561996855115,-3.283605931175963,
 -6.125848009909261];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p40.svg"}], p40];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.646514,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.349305,0,1.102851,0,1.005677,0,1.133327,0,1.123901,0,0.552555,0,0.391315,0,1.411809,0,0.489586,0,1.417820];
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
correctResults=List[ 3.98525951367615 ,-5.024001666359878,-1.329739300383345,
 -5.381390770624149];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p41.svg"}], p41];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.978147,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.302007,0,0.432242,0,0.074013,0,0.683485,0.273650,0,1.001293,0,0.295182,0,0,0.114225,0.489279,0,0,0.096726];
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
correctResults=List[-2.824751297373469,-1.7619898790632  ,-1.299325492073292,
  1.428620040425468];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p42.svg"}], p42];

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
  WriteString["VIRUSReduced_convergence_times.csv", "3.762102,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.228723,0,0.999942,0,1.406028,0,0.556798,0,0.718562,0,0.642790,0,0.471826,0,0.475043,0,0.169330,0,1.755433];
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
correctResults=List[ 1.286810097109068,-2.608216414256676,-0.661330162847065,
 -4.103587692838191];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p43.svg"}], p43];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.424476,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.394086,0,0.778802,0,0.961449,0,0.283552,0,0.957609,0,0.490662,0,0.762884,0,0.914931,0,0.831903,0,0.038011];
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
correctResults=List[ 1.542337875187379,-3.021828838639235,-0.735435960663573,
 -2.93846913213538 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p44.svg"}], p44];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.413888,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.106860,0,0,0.049740,0,0.961449,0,1.399122,0,3.785969,0,1.456198,0.190290,0,0,0.765847,0,1.248335,0,0.581127];
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
correctResults=List[ 6.271790596961957,-5.833722365103935,-4.197359355015771,
 -7.667921853909959];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p45.svg"}], p45];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.544937,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.331573,0,0.685352,0,0.848006,0,0.372752,0,0.364499,0,0.812508,0,0.461921,0,0.299918,0,1.199256,0,1.305894];
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
correctResults=List[ 3.942167680084688,-4.235141553931212,-2.512831358088271,
 -5.338298937032689];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p46.svg"}], p46];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.681678,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.367678,0,0.697081,0,1.166219,0,0.864145,0,0.228692,0,0.431049,0,0.387252,0,0.930593,0,0.315823,0,1.544425];
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
correctResults=List[ 2.319204386642566,-3.686409229441269,-0.615531637195969,
 -3.715335643590567];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p47.svg"}], p47];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.932958,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.005479,0,0.695591,0,0.833239,0,0.627874,0,1.510681,0,1.570060,0,2.062605,0,1.003140,0,0.110770,0,1.525284,0];
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
correctResults=List[-12.883041478310528,  5.345440530679185, -9.846285021948145,
  -4.239281249184451];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p48.svg"}], p48];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.944724,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.886578,0,0.761800,0,1.359956,0,0.392565,0,1.228446,0,2.276622,0,1.441887,0,0.623450,0,0.946362,0,1.163818,0];
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
correctResults=List[-11.653943746899069,  4.234419712326486, -8.218711874532028,
  -3.644812759895167];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p49.svg"}], p49];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.081484,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.044347,0,0.606113,0,1.334109,0,1.735314,0,1.027740,0,1.287305,0,1.201117,0,1.256580,0,1.526890,0,0.631711,0];
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
correctResults=List[-7.079084894378622, 4.663599112532098,-9.005040839257026,
 -3.716842595581393];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p50.svg"}], p50];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.651226,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.420479,0,0.534607,0,1.167248,0,1.225855,0,1.335152,0,2.072373,0,1.592750,0,1.111838,0,1.526890,0,1.703265,0];
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
correctResults=List[-12.81508853993267 ,  7.092869529030282,-12.409876904534624,
  -5.170547363300166];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p51.svg"}], p51];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.690456,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.446456,0,0.276767,0,0.943522,0,1.400191,0,1.795228,0,1.074914,0,1.290517,0,1.861024,0,1.551678,0,1.361982,0];
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
correctResults=List[-14.698608576431575,  6.604321369430978, -8.672352246529979,
  -7.930975702106201];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p52.svg"}], p52];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.002278,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.065255,0,0,0.834986,0,0.989881,0,0.952894,0,0.775148,0,0.651585,0,0.813170,0,0.803024,0,1.150424,0,0.217826];
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
correctResults=List[ 4.803912732124311,-4.938096017591929,-3.251383779419676,
 -6.200043989072312];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p53.svg"}], p53];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.254192,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.998088,0,1.226591,0,1.360244,0,1.101652,0,0.541797,0,0,0.709243,0.275626,0,1.700242,0,0.216615,0,0.884920,0];
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
correctResults=List[-5.075401066913904, 4.586529651186425,-4.904600895531101,
 -7.663143616615973];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p54.svg"}], p54];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.015018,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.586678,0,2.544729,0,0.673845,0,1.978301,0,1.001642,0,0,0.903718,1.084797,0,1.174302,0,0,0.128925,1.068406,0];
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
correctResults=List[ -6.482582770405799,  9.059142479202707, -9.596732171496482,
 -11.916237996683154];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p55.svg"}], p55];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.145343,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.065255,0,0,0.909708,0,0.621121,0,0.774493,0,0.752282,0,1.005680,0,0.788028,0,1.026649,0,1.346245,0,0.038011];
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
correctResults=List[ 5.819136652469617,-6.099466831438517,-3.820369539631621,
 -7.215267909417618];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p56.svg"}], p56];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.327472,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.190702,0,0.570360,0,1.445540,0,1.314604,0,1.203272,0,1.350176,0,1.174958,0,1.600791,0,1.526890,0,0.453730,0];
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
correctResults=List[-8.502669095173182, 4.568703302643258,-9.042887799031659,
 -3.489204016029082];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p57.svg"}], p57];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.831023,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.138107,0,1.068067,0,1.263746,0,0.931607,0,0.333700,0,1.487970,0,2.610433,0,1.478505,0,0.466474,0,1.163818,0];
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
correctResults=List[-11.720351934050024,  9.871225812933405,-14.98988633518152 ,
  -8.147250500459512];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p58.svg"}], p58];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.942428,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.080889,0,0.629358,0,0.848006,0,0.752136,0,0.843513,0,0.201066,0,0.788028,0,0.318411,0,0.880983,0,0.399476];
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
correctResults=List[ 2.236120540022832,-3.153911368005251,-1.663791654480712,
 -3.632251796970833];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p59.svg"}], p59];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.741866,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.540231,0,0.517180,0,0.422382,0,0.618223,0.091651,0,0,1.584871,0,0.512209,0,0.579400,0,1.762678,0.143639,0];
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
correctResults=List[ 8.329057733833771,-7.914052643828368,-5.277978775872008,
 -9.725188990781772];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p60.svg"}], p60];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.772464,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.810338,0,0.051088,0,0,0.148397,0.567353,0,0.858445,0,0,0.670153,0,0.558178,0,0.528636,0.929507,0,0.323456,0];
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
correctResults=List[-2.963423281697279,-1.761989879063199, 2.742676680765806,
 -2.613382132413631];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p61.svg"}], p61];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.445549,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.307489,0,1.084501,0,0.678848,0,0.671291,0,0.815337,0,0.797849,0,0.391315,0,1.220833,0,0.824714,0,1.456352];
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
correctResults=List[ 4.452878485739214,-5.527436860852217,-1.739138511790053,
 -5.849009742687214];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p62.svg"}], p62];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.248528,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.263977,0,1.374333,0,1.039446,0,0.509768,0,0.810868,0,1.375259,0,1.788057,0,0.556080,0,0.466474,0,0.965653,0];
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
correctResults=List[-9.223125278566778, 5.950558671257455,-9.490534923123644,
 -5.805267629165491];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p63.svg"}], p63];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.149916,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.244554,0,1.231108,0,1.259845,0,0.209255,0,0.197974,0,0.347982,0,1.336616,0,0.439565,0.117711,0,0,0.929750];
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
correctResults=List[ 1.178031896065463,-2.925820680363758,-0.234947695696376,
 -3.229711739581107];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p64.svg"}], p64];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.314360,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.253212,0,0.983485,0,0.588094,1.468390,0,0.123755,0,1.687657,0,0,0.344075,0,0.344831,1.559112,0,0,0.755438];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p65.svg"}], p65];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.108050,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.298416,0,0.055604,0.276079,0,0,0.616416,0.351484,0,1.429334,0,0.656593,0,0,0.119131,0.371786,0,0,0.019662];
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
correctResults=List[-3.693534644682105,-1.7619898790632  ,-2.168108839381928,
  2.297403387734104];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p66.svg"}], p66];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.194506,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.276024,0,0.031982,0,0.515887,0,0,0.899825,0.026059,0,0.766099,0,0,0.405030,0,1.040613,0.654116,0,1.244551,0];
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
correctResults=List[-3.328320718590176,-1.761989879063199, 1.463841089065537,
 -1.334546540713361];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p67.svg"}], p67];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.860186,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.644559,0,0.760074,0,0.734563,0,1.287565,0,0.912109,0,0.587412,0,1.339918,0,1.287637,0,1.125884,0.143639,0];
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
correctResults=List[ 6.14391336899994 ,-6.846981086783384,-3.710523186100737,
 -7.540044625947941];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p68.svg"}], p68];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.823362,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.579714,0,1.304317,0,0.642661,0.415825,0,0.243163,0,0,0.675365,0,1.953017,0,0.352757,0.243385,0,0,0.656356];
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
correctResults=List[ 0.331777673025708,-3.061970421128248, 0.596305357006178,
 -2.587190260400644];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p69.svg"}], p69];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.066561,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.232307,0,0,0.498452,0,0.479247,0,0.283552,0.319612,0,0.152703,0,0,0.487065,0.035613,0,0,1.321707,0.143639,0];
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
correctResults=List[ 0.601942725397775,-2.047003330290737,-0.822689326329247,
 -1.998073982345776];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p70.svg"}], p70];

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
  WriteString["VIRUSReduced_convergence_times.csv", "3.953896,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.572114,0,0.940566,0,0.655177,0,1.493908,0,1.407442,0,1.901027,0,1.052542,0,1.298663,0,1.628520,0,1.760145,0];
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
correctResults=List[-12.120478003397288,  8.014076837799509,-12.757924221045895,
  -6.664914664327346];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p71.svg"}], p71];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.710105,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.387319,0,0.619332,0,0.992466,0,0.402786,0,0.718562,0,0.956817,0,1.033622,0,0.958523,0,0.453642,0,1.386627];
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
correctResults=List[ 3.065162102997286,-4.509570219569182,-1.217369494808928,
 -4.461293359945286];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p72.svg"}], p72];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.909697,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.023439,0,0,0.647897,0.400147,0,0,0.171994,0.068786,0,0,1.295276,0,0.210993,0.240742,0,0,1.493237,0,1.124244];
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
correctResults=List[ 5.529951918244344,-5.991882503979181,-3.917091471923606,
 -6.926083175192345];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p73.svg"}], p73];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.676754,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.227034,0,0.797530,0,1.018026,0,0.774493,0,2.189329,0,1.295276,0,0.361601,0,0.243870,0,1.027972,0,0.217826];
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
correctResults=List[ 6.157389263927919,-5.398891218953108,-4.007188884390609,
 -7.55352052087592 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p74.svg"}], p74];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.152955,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.886578,0,1.119143,0,1.135657,0,0.111414,0,0.781073,0,1.826104,0,1.788057,0,1.208647,0,1.148135,0,1.163818,0];
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
correctResults=List[-12.711192309473457,  7.381608927152116,-12.428392327474878,
  -5.729510736603579];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p75.svg"}], p75];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.168626,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.059981,0,1.152603,0,1.103323,0,0.752136,0,0.569823,0,1.906717,0,0.963779,0,0.821706,0,1.468697,0,0.217826];
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
correctResults=List[  9.318595360775571, -8.163411074333167, -5.950565649482622,
 -11.706039806117884];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p76.svg"}], p76];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.016592,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.796610,0,0.695591,0,1.000100,0,1.137332,0,1.708156,0,0.973279,0,1.514271,0,2.342618,0,1.184325,0,1.525284,0];
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
correctResults=List[-16.892086773290934,  8.969083675644358,-11.299207113273068,
 -10.03364544778987 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p77.svg"}], p77];

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
  WriteString["VIRUSReduced_convergence_times.csv", "12.877565,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.416683,0,0.359601,0.304798,0,0,0.310651,0,0.111825,0.653387,0,0.133400,0,0.097699,0,0.678904,0,0,0.349936];
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
correctResults=List[-1.630143316824133,-1.7619898790632  ,-0.104717511523957,
  0.234012059876132];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p78.svg"}], p78];

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
  WriteString["VIRUSReduced_convergence_times.csv", "3.416885,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.310665,0,0.479724,0,0.479247,0,1.711662,0,0.638420,0,0.877007,0,1.139277,0,0.448999,0,0.611541,0,0.217826];
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
correctResults=List[ 6.485365282118303,-6.68611220365045 ,-4.466869292629974,
 -7.881496539066304];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p79.svg"}], p79];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.914368,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.237789,0,0.319710,0,0,0.090671,0,0.087310,0,0.574905,0,0.147969,0,0.430682,0,0.452775,0,0.447447,0.139970,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p80.svg"}], p80];

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
  WriteString["VIRUSReduced_convergence_times.csv", "3.929226,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.802735,0,0.685186,0,1.135657,0,1.728087,0,1.735640,0,1.600681,0,1.701452,0,1.433592,0,1.072532,0,1.561982,0];
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
correctResults=List[-13.349475412741008,  7.929913648865182,-11.74714615154974 ,
  -7.507366355954849];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p81.svg"}], p81];

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
  WriteString["VIRUSReduced_convergence_times.csv", "13.457542,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.259120,0,0.242906,0,0,0.292857,0.251426,0,0,0.389904,0,0.213119,0.615957,0,0,0.263308,0,0.547836,0.508776,0];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p82.svg"}], p82];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.585208,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.504111,0,0.480882,0,0.805381,0,1.668922,0,1.510681,0,1.130292,0,1.044413,0,1.637024,0,1.769807,0,1.345469,0];
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
correctResults=List[-12.441218345537486,  7.144164571068246, -9.378529570586606,
  -8.30448478132411 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p83.svg"}], p83];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.896982,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.253222,0.276578,0,1.006706,0,0,0.165897,0.690768,0,1.405228,0,0.309153,0,0.206775,0,0.852913,0,0,0.806814];
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
correctResults=List[-2.704313265521224,-1.7619898790632  ,-1.178887460221047,
  1.308182008573223];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p84.svg"}], p84];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.974054,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.655957,0,0.838414,0,1.231867,0,1.048583,0,1.258472,0,1.713392,0,1.485063,0,0.871042,0,0.845230,0,0.765656,0];
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
correctResults=List[-8.765936642502249, 4.135345587036569,-8.51312150721483 ,
 -3.152254876632532];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p85.svg"}], p85];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.713676,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.461873,0,0,0.012473,0,0.337372,0,1.354634,0,3.261454,0,0.522912,0.165147,0,0.557590,0,0,1.297167,0,0.038011];
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
correctResults=List[ 4.404000579684955,-4.232780928738689,-3.650790229288952,
 -5.800131836632956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p86.svg"}], p86];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.008633,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.821960,0,0.411055,0.116110,0,0,0.478664,0.102046,0,0.421450,0,0,0.512716,0,0.617709,0.778303,0,0,0.096726];
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
correctResults=List[-1.07705316915005 ,-1.7619898790632  , 0.448372636150127,
 -0.319078087797951];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p87.svg"}], p87];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.356737,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.410543,0,1.399871,0,1.039446,0,0.790918,0,0.840661,0,1.938814,0,1.961269,0,1.951041,0,0.744345,0,1.163818,0];
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
correctResults=List[-13.439245193813742, 10.634543368122538,-16.038652203130052,
  -8.625119742889245];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p88.svg"}], p88];

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
  WriteString["VIRUSReduced_convergence_times.csv", "13.240726,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.513611,0,1.190247,0,1.323316,0,1.436157,0,0.412770,0,1.521675,0,1.220803,0,0.842464,0,0.427864,0,0.834337];
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
correctResults=List[  9.262123238106348, -8.208008029513335, -5.144673077675714,
 -10.841755082120764];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p89.svg"}], p89];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.723244,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.485945,0,1.175492,0,1.761289,0.675974,0,0.032295,0,0,0.466230,0,1.164421,0,1.245553,1.849129,0,0,2.128092];
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
correctResults=List[-2.679244629827981,-1.7619898790632  , 2.458498028896509,
 -2.329203480544333];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p90.svg"}], p90];

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
  WriteString["VIRUSReduced_convergence_times.csv", "11.984420,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.489749,0,0.223800,0,0.515887,0,0.318946,0,0.581061,0,0.439693,0,0,0.762375,0,0.073652,0.278584,0,0.323456,0];
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
correctResults=List[-0.943496582182749,-1.761989879063199, 0.722749981251276,
 -0.5934554328991  ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p91.svg"}], p91];

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
  WriteString["VIRUSReduced_convergence_times.csv", "4.007204,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[1.187315,0,2.393961,0,0.931172,0,1.596207,0,0.495835,0,0.782712,0,0.124002,0,0.892365,0,0.488040,0,1.250057,0];
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
correctResults=List[-5.284721425493201, 7.194359668991525,-8.672456860484864,
 -9.11094768727241 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p92.svg"}], p92];

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
  WriteString["VIRUSReduced_convergence_times.csv", "10.141665,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.429346,0,0.657544,0,1.433599,0,0.952442,0,0.883239,0,0.736932,0,0.391569,0,1.383880,0,0.892880,0,0.834337];
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
correctResults=List[ 4.423230002939683,-5.044641443543409,-1.726188985480086,
 -5.819361259887684];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p93.svg"}], p93];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.595769,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,1.336210,0,0.823257,0,1.690352,0.078444,0,0,0.195666,0,0.132332,0,0.021525,0,0.905873,0.997177,0,0,0.087551];
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
correctResults=List[-1.331677142319923,-1.7619898790632  , 1.110930541388451,
 -0.981635993036275];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p94.svg"}], p94];

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
  WriteString["VIRUSReduced_convergence_times.csv", "6.268389,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0.253004,0,0,0.909708,0,0.621121,0,0.997606,0,0.387362,0,1.230777,0,0.411887,0,0.150647,0,1.591148,0,0.581127];
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
correctResults=List[ 8.071917574663738,-6.921805360369658,-5.567785172078028,
 -9.745394398679604];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p95.svg"}], p95];

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
  WriteString["VIRUSReduced_convergence_times.csv", "7.134387,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.695879,0,1.074854,0.021910,0,0,0.346782,0,0.525709,0,1.237291,0,1.598467,0,0.621671,1.673137,0,0,1.254519];
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
correctResults=List[-1.890466854710235,-2.117942344025197, 1.669720253778763,
 -2.252330635350582];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p96.svg"}], p96];

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
  WriteString["VIRUSReduced_convergence_times.csv", "9.050219,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.310665,0,0.386274,0,1.018026,0,1.265208,0,2.850803,0,0.973431,0,0.637673,0.166201,0,0,0.538170,0,0.399476];
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
correctResults=List[ 5.141855861365777,-4.705474981121992,-3.640211475862281,
 -6.537987118313779];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p97.svg"}], p97];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.545927,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.215418,0.185399,0,0.544607,0,1.696021,0,0.742734,0,0.341640,0,0.054414,0,0.286600,0,0.579009,0,0,0.412320];
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
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p98.svg"}], p98];

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
  WriteString["VIRUSReduced_convergence_times.csv", "5.058163,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

rsys=VIRUSReduced[0,0.059981,0,0.610630,0,1.046459,0,1.845350,0,2.326289,0,1.263026,0,0.336457,0.277919,0,0,0.807611,0.143639,0];
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
correctResults=List[ 8.567917640626412,-6.627852791004156,-5.953424272907989,
 -9.964048897574413];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol,(hL1N3Spos[t] - hL1N3Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2","y3"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/virus", "VIRUSReduced_p99.svg"}], p99];

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
  WriteString["VIRUSReduced_convergence_times.csv", "8.717360,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t] + hL1N2Spos[t] + hL1N2Sneg[t] + hL1N3Spos[t] + hL1N3Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["VIRUSReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["VIRUSReduced_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
