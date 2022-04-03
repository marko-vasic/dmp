#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];

HEART[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x0Sneg, x0SnegInitial],
conc[x1Spos, x1SposInitial],
conc[x1Sneg, x1SnegInitial],
rxn[x0Spos,iL0N0Spos+iL0N2Spos+iL0N3Spos+iL0N4Sneg+iL0N7Sneg,1],
rxn[x0Sneg,iL0N0Sneg+iL0N2Sneg+iL0N3Sneg+iL0N4Spos+iL0N7Spos,1],
rxn[x1Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Sneg+iL0N4Sneg+iL0N6Sneg,1],
rxn[x1Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Spos+iL0N4Spos+iL0N6Spos,1],
conc[iL0N0Sneg,2.2061376334],
conc[iL0N1Sneg,0.5232925276],
conc[iL0N2Sneg,0.6654773492],
conc[iL0N3Sneg,2.2042284948],
conc[iL0N4Sneg,2.2514632358],
conc[iL0N5Sneg,1.4409474327],
conc[iL0N6Sneg,2.2612309931],
conc[iL0N7Sneg,2.2339390967],
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
rxn[hL0N5Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N5Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N6Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N6Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N7Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N7Sneg,hL1N0Spos+hL1N1Sneg,1],
conc[hL1N0Spos,0.9311815160],
conc[hL1N1Sneg,0.9009135303]
}

cnt = Count[HEART[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[HEARTReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial], rxn[___]]
Print["Number of reactions (reduced form): " <> ToString[cntReduced]];

tmax=30;
totalCount=0;
correctCount=0;
matchingCount=0;

Print["Example Index (i)"];
Print["Expected Output (E)"];
Print["CRN Output (C)"];
Print["NN Output (N)"];

Print["i: E C N"];

rsys=HEART[0,1.720147,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 1, correctCount+=1, warning = "*"];
If[predicted == 1, matchingCount+=1, warning = "**"];
Print["8: 1 " <> ToString[predicted] <> " 1 " <> warning];
plotter={((hL1N1Spos[t] - hL1N1Sneg[t])-(hL1N0Spos[t] - hL1N0Sneg[t]))/.sol};
p8=Plot[plotter, {t, 0, tmax},
        (* Make plot lines thicker *)
        PlotStyle->{RGBColor["#005f86"], Thickness[0.03]},
        PlotRange->All,
        AxesLabel -> {Style["time", Bold, 20], Style["value", Bold, 20]},
        (* Numbers on x and y axis *)
        LabelStyle -> {FontSize->20},
        Ticks -> {{}, {-1.83, 0.34}},
        GridLines -> {{}, { {-1.83, Directive[Blue, Thickness[0.02], Dashed]}, {0.34, Directive[Blue, Thickness[0.02], Dashed]} }}
        ];
Print["y1(0) - y0(0) [start value]: " <> ToString[((hL1N1Spos[0] - hL1N1Sneg[0])-(hL1N0Spos[0] - hL1N0Sneg[0]))/.sol]];
Print["y1(tmax) - y0(tmax) [final value]: " <> ToString[((hL1N1Spos[tmax] - hL1N1Sneg[tmax])-(hL1N0Spos[tmax] - hL1N0Sneg[tmax]))/.sol]];

rsys=HEART[0.310582,0,0,1.620184];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
If[predicted == 0, correctCount+=1, warning = "*"];
If[predicted == 0, matchingCount+=1, warning = "**"];
Print["13: 0 " <> ToString[predicted] <> " 0 " <> warning];
plotter={((hL1N1Spos[t] - hL1N1Sneg[t])-(hL1N0Spos[t] - hL1N0Sneg[t]))/.sol};
p13=Plot[plotter, {t, 0, tmax},
        (* Make plot lines thicker *)
        PlotStyle->{RGBColor["#00a9b7"], Thickness[0.03]},
        PlotRange->All,
        AxesLabel -> {Style["time", Bold, 20], Style["value", Bold, 20]},
        (* Numbers on x and y axis *)
        LabelStyle -> {FontSize->20},
        Ticks -> {{}, {-1.83, 0.34}},
        GridLines -> {{}, { {-1.83, Directive[Blue, Thickness[0.02], Dashed]}, {0.34, Directive[Blue, Thickness[0.02], Dashed]} }}
        ];
Print["y1(0) - y0(0) [start value]: " <> ToString[((hL1N1Spos[0] - hL1N1Sneg[0])-(hL1N0Spos[0] - hL1N0Sneg[0]))/.sol]];
Print["y1(tmax) - y0(tmax) [final value]: " <> ToString[((hL1N1Spos[tmax] - hL1N1Sneg[tmax])-(hL1N0Spos[tmax] - hL1N0Sneg[tmax]))/.sol]];

s=Show[p8,p13];
Export["./HEART-custom.pdf", s];