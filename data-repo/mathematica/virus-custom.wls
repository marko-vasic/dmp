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

tmax=30;

rsys=VIRUS[10.832427,0,8.891756,0,11.042110,0,14.558414,0,19.666872,0,9.937790,0,20.009293,0,16.920303,0,21.814613,0,19.091640,0];

sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic,
                   AccuracyGoal->Automatic,
                   Method->{"EquationSimplification"->"Residual"}];

plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,
         (hL1N1Spos[t] - hL1N1Sneg[t])/.sol,
         (hL1N2Spos[t] - hL1N2Sneg[t])/.sol,
         (hL1N3Spos[t] - hL1N3Sneg[t])/.sol};

p73=Plot[plotter, {t, 0, tmax},
         PlotLabels->{"H1N1","H3N2","HRV","RSV"},
         PlotRange->All,
         AxesLabel -> {Style["time", Bold, 20], Style["value", Bold, 20]},
         LabelStyle -> {FontSize->14},
         PlotStyle->{{RGBColor["#bf5700"], Thickness[0.015]}, {RGBColor["#579d42"], Thickness[0.015]},
                     {RGBColor["#005f86"], Thickness[0.015]}, {RGBColor["#333f48"], Thickness[0.015]}}
        ];

Export["./VIRUS_custom.pdf", p73];