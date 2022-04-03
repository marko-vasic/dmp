#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];
Get[Directory[] <> "/utils.m"];

MNIST[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_,x10SposInitial_,x10SnegInitial_,x11SposInitial_,x11SnegInitial_,x12SposInitial_,x12SnegInitial_,x13SposInitial_,x13SnegInitial_,x14SposInitial_,x14SnegInitial_,x15SposInitial_,x15SnegInitial_,x16SposInitial_,x16SnegInitial_,x17SposInitial_,x17SnegInitial_,x18SposInitial_,x18SnegInitial_,x19SposInitial_,x19SnegInitial_,x20SposInitial_,x20SnegInitial_,x21SposInitial_,x21SnegInitial_,x22SposInitial_,x22SnegInitial_,x23SposInitial_,x23SnegInitial_,x24SposInitial_,x24SnegInitial_,x25SposInitial_,x25SnegInitial_,x26SposInitial_,x26SnegInitial_,x27SposInitial_,x27SnegInitial_,x28SposInitial_,x28SnegInitial_,x29SposInitial_,x29SnegInitial_,x30SposInitial_,x30SnegInitial_,x31SposInitial_,x31SnegInitial_,x32SposInitial_,x32SnegInitial_,x33SposInitial_,x33SnegInitial_,x34SposInitial_,x34SnegInitial_,x35SposInitial_,x35SnegInitial_,x36SposInitial_,x36SnegInitial_,x37SposInitial_,x37SnegInitial_,x38SposInitial_,x38SnegInitial_,x39SposInitial_,x39SnegInitial_,x40SposInitial_,x40SnegInitial_,x41SposInitial_,x41SnegInitial_,x42SposInitial_,x42SnegInitial_,x43SposInitial_,x43SnegInitial_,x44SposInitial_,x44SnegInitial_,x45SposInitial_,x45SnegInitial_,x46SposInitial_,x46SnegInitial_,x47SposInitial_,x47SnegInitial_,x48SposInitial_,x48SnegInitial_,x49SposInitial_,x49SnegInitial_,x50SposInitial_,x50SnegInitial_,x51SposInitial_,x51SnegInitial_,x52SposInitial_,x52SnegInitial_,x53SposInitial_,x53SnegInitial_,x54SposInitial_,x54SnegInitial_,x55SposInitial_,x55SnegInitial_,x56SposInitial_,x56SnegInitial_,x57SposInitial_,x57SnegInitial_,x58SposInitial_,x58SnegInitial_,x59SposInitial_,x59SnegInitial_,x60SposInitial_,x60SnegInitial_,x61SposInitial_,x61SnegInitial_,x62SposInitial_,x62SnegInitial_,x63SposInitial_,x63SnegInitial_] :=
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
conc[x10Spos, x10SposInitial],
conc[x10Sneg, x10SnegInitial],
conc[x11Spos, x11SposInitial],
conc[x11Sneg, x11SnegInitial],
conc[x12Spos, x12SposInitial],
conc[x12Sneg, x12SnegInitial],
conc[x13Spos, x13SposInitial],
conc[x13Sneg, x13SnegInitial],
conc[x14Spos, x14SposInitial],
conc[x14Sneg, x14SnegInitial],
conc[x15Spos, x15SposInitial],
conc[x15Sneg, x15SnegInitial],
conc[x16Spos, x16SposInitial],
conc[x16Sneg, x16SnegInitial],
conc[x17Spos, x17SposInitial],
conc[x17Sneg, x17SnegInitial],
conc[x18Spos, x18SposInitial],
conc[x18Sneg, x18SnegInitial],
conc[x19Spos, x19SposInitial],
conc[x19Sneg, x19SnegInitial],
conc[x20Spos, x20SposInitial],
conc[x20Sneg, x20SnegInitial],
conc[x21Spos, x21SposInitial],
conc[x21Sneg, x21SnegInitial],
conc[x22Spos, x22SposInitial],
conc[x22Sneg, x22SnegInitial],
conc[x23Spos, x23SposInitial],
conc[x23Sneg, x23SnegInitial],
conc[x24Spos, x24SposInitial],
conc[x24Sneg, x24SnegInitial],
conc[x25Spos, x25SposInitial],
conc[x25Sneg, x25SnegInitial],
conc[x26Spos, x26SposInitial],
conc[x26Sneg, x26SnegInitial],
conc[x27Spos, x27SposInitial],
conc[x27Sneg, x27SnegInitial],
conc[x28Spos, x28SposInitial],
conc[x28Sneg, x28SnegInitial],
conc[x29Spos, x29SposInitial],
conc[x29Sneg, x29SnegInitial],
conc[x30Spos, x30SposInitial],
conc[x30Sneg, x30SnegInitial],
conc[x31Spos, x31SposInitial],
conc[x31Sneg, x31SnegInitial],
conc[x32Spos, x32SposInitial],
conc[x32Sneg, x32SnegInitial],
conc[x33Spos, x33SposInitial],
conc[x33Sneg, x33SnegInitial],
conc[x34Spos, x34SposInitial],
conc[x34Sneg, x34SnegInitial],
conc[x35Spos, x35SposInitial],
conc[x35Sneg, x35SnegInitial],
conc[x36Spos, x36SposInitial],
conc[x36Sneg, x36SnegInitial],
conc[x37Spos, x37SposInitial],
conc[x37Sneg, x37SnegInitial],
conc[x38Spos, x38SposInitial],
conc[x38Sneg, x38SnegInitial],
conc[x39Spos, x39SposInitial],
conc[x39Sneg, x39SnegInitial],
conc[x40Spos, x40SposInitial],
conc[x40Sneg, x40SnegInitial],
conc[x41Spos, x41SposInitial],
conc[x41Sneg, x41SnegInitial],
conc[x42Spos, x42SposInitial],
conc[x42Sneg, x42SnegInitial],
conc[x43Spos, x43SposInitial],
conc[x43Sneg, x43SnegInitial],
conc[x44Spos, x44SposInitial],
conc[x44Sneg, x44SnegInitial],
conc[x45Spos, x45SposInitial],
conc[x45Sneg, x45SnegInitial],
conc[x46Spos, x46SposInitial],
conc[x46Sneg, x46SnegInitial],
conc[x47Spos, x47SposInitial],
conc[x47Sneg, x47SnegInitial],
conc[x48Spos, x48SposInitial],
conc[x48Sneg, x48SnegInitial],
conc[x49Spos, x49SposInitial],
conc[x49Sneg, x49SnegInitial],
conc[x50Spos, x50SposInitial],
conc[x50Sneg, x50SnegInitial],
conc[x51Spos, x51SposInitial],
conc[x51Sneg, x51SnegInitial],
conc[x52Spos, x52SposInitial],
conc[x52Sneg, x52SnegInitial],
conc[x53Spos, x53SposInitial],
conc[x53Sneg, x53SnegInitial],
conc[x54Spos, x54SposInitial],
conc[x54Sneg, x54SnegInitial],
conc[x55Spos, x55SposInitial],
conc[x55Sneg, x55SnegInitial],
conc[x56Spos, x56SposInitial],
conc[x56Sneg, x56SnegInitial],
conc[x57Spos, x57SposInitial],
conc[x57Sneg, x57SnegInitial],
conc[x58Spos, x58SposInitial],
conc[x58Sneg, x58SnegInitial],
conc[x59Spos, x59SposInitial],
conc[x59Sneg, x59SnegInitial],
conc[x60Spos, x60SposInitial],
conc[x60Sneg, x60SnegInitial],
conc[x61Spos, x61SposInitial],
conc[x61Sneg, x61SnegInitial],
conc[x62Spos, x62SposInitial],
conc[x62Sneg, x62SnegInitial],
conc[x63Spos, x63SposInitial],
conc[x63Sneg, x63SnegInitial],
rxn[x0Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x0Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x1Spos,iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x1Sneg,iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x2Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x2Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x3Spos,iL0N0Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x3Sneg,iL0N0Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x4Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x4Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x5Spos,iL0N0Spos+iL0N1Sneg+iL0N3Spos,1],
rxn[x5Sneg,iL0N0Sneg+iL0N1Spos+iL0N3Sneg,1],
rxn[x6Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x6Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x7Spos,iL0N0Spos+iL0N1Spos+iL0N3Spos,1],
rxn[x7Sneg,iL0N0Sneg+iL0N1Sneg+iL0N3Sneg,1],
rxn[x8Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Sneg,1],
rxn[x8Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Spos,1],
rxn[x9Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Sneg,1],
rxn[x9Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Spos,1],
rxn[x10Spos,iL0N0Sneg+iL0N1Sneg+iL0N3Spos,1],
rxn[x10Sneg,iL0N0Spos+iL0N1Spos+iL0N3Sneg,1],
rxn[x11Spos,iL0N0Spos+iL0N1Sneg+iL0N3Sneg,1],
rxn[x11Sneg,iL0N0Sneg+iL0N1Spos+iL0N3Spos,1],
rxn[x12Spos,iL0N0Spos+iL0N1Sneg,1],
rxn[x12Sneg,iL0N0Sneg+iL0N1Spos,1],
rxn[x13Spos,iL0N0Sneg+iL0N1Sneg+iL0N2Spos,1],
rxn[x13Sneg,iL0N0Spos+iL0N1Spos+iL0N2Sneg,1],
rxn[x14Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x14Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
rxn[x15Spos,iL0N0Spos+iL0N1Spos+iL0N2Sneg+iL0N3Spos,1],
rxn[x15Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Spos+iL0N3Sneg,1],
rxn[x16Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x16Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x17Spos,iL0N0Sneg+iL0N1Spos+iL0N3Spos,1],
rxn[x17Sneg,iL0N0Spos+iL0N1Sneg+iL0N3Sneg,1],
rxn[x18Spos,iL0N0Sneg+iL0N3Spos,1],
rxn[x18Sneg,iL0N0Spos+iL0N3Sneg,1],
rxn[x19Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x19Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x20Spos,iL0N0Sneg,1],
rxn[x20Sneg,iL0N0Spos,1],
rxn[x22Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x22Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x23Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x23Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x24Spos,iL0N0Spos+iL0N1Sneg+iL0N2Spos+iL0N3Spos,1],
rxn[x24Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Sneg+iL0N3Sneg,1],
rxn[x25Spos,iL0N0Spos+iL0N1Spos+iL0N2Sneg+iL0N3Spos,1],
rxn[x25Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Spos+iL0N3Sneg,1],
rxn[x26Spos,iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x26Sneg,iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x27Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x27Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x28Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x28Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
rxn[x29Spos,iL0N1Spos+iL0N2Sneg+iL0N3Spos,1],
rxn[x29Sneg,iL0N1Sneg+iL0N2Spos+iL0N3Sneg,1],
rxn[x30Spos,iL0N0Spos+iL0N2Sneg+iL0N3Sneg,1],
rxn[x30Sneg,iL0N0Sneg+iL0N2Spos+iL0N3Spos,1],
rxn[x31Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x31Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x32Spos,iL0N0Sneg+iL0N2Spos+iL0N3Spos,1],
rxn[x32Sneg,iL0N0Spos+iL0N2Sneg+iL0N3Sneg,1],
rxn[x33Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Spos,1],
rxn[x33Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Sneg,1],
rxn[x34Spos,iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x34Sneg,iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x35Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x35Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
rxn[x36Spos,iL0N0Sneg+iL0N1Spos+iL0N3Spos,1],
rxn[x36Sneg,iL0N0Spos+iL0N1Sneg+iL0N3Sneg,1],
rxn[x37Spos,iL0N1Spos+iL0N2Sneg,1],
rxn[x37Sneg,iL0N1Sneg+iL0N2Spos,1],
rxn[x39Spos,iL0N0Spos+iL0N1Spos+iL0N3Spos,1],
rxn[x39Sneg,iL0N0Sneg+iL0N1Sneg+iL0N3Sneg,1],
rxn[x40Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x40Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x41Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x41Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x42Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x42Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x43Spos,iL0N0Spos+iL0N1Sneg+iL0N2Spos,1],
rxn[x43Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Sneg,1],
rxn[x44Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x44Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
rxn[x45Spos,iL0N0Spos+iL0N1Spos,1],
rxn[x45Sneg,iL0N0Sneg+iL0N1Sneg,1],
rxn[x46Spos,iL0N0Sneg+iL0N3Sneg,1],
rxn[x46Sneg,iL0N0Spos+iL0N3Spos,1],
rxn[x47Spos,iL0N0Sneg+iL0N1Spos+iL0N2Sneg,1],
rxn[x47Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Spos,1],
rxn[x48Spos,iL0N0Spos+iL0N1Sneg+iL0N3Spos,1],
rxn[x48Sneg,iL0N0Sneg+iL0N1Spos+iL0N3Sneg,1],
rxn[x49Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x49Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x50Spos,iL0N0Spos+iL0N1Sneg+iL0N2Spos,1],
rxn[x50Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Sneg,1],
rxn[x51Spos,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x51Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x52Spos,iL0N1Sneg+iL0N2Spos,1],
rxn[x52Sneg,iL0N1Spos+iL0N2Sneg,1],
rxn[x53Spos,iL0N0Spos+iL0N1Spos+iL0N2Sneg,1],
rxn[x53Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Spos,1],
rxn[x54Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x54Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x55Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos,1],
rxn[x55Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg,1],
rxn[x56Spos,iL0N0Spos+iL0N1Spos+iL0N2Sneg+iL0N3Spos,1],
rxn[x56Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Spos+iL0N3Sneg,1],
rxn[x57Spos,iL0N0Spos+iL0N1Sneg+iL0N2Spos+iL0N3Spos,1],
rxn[x57Sneg,iL0N0Sneg+iL0N1Spos+iL0N2Sneg+iL0N3Sneg,1],
rxn[x58Spos,iL0N0Spos+iL0N1Spos+iL0N2Sneg+iL0N3Spos,1],
rxn[x58Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Spos+iL0N3Sneg,1],
rxn[x59Spos,iL0N0Spos+iL0N1Spos+iL0N3Sneg,1],
rxn[x59Sneg,iL0N0Sneg+iL0N1Sneg+iL0N3Spos,1],
rxn[x60Spos,iL0N0Sneg+iL0N2Spos,1],
rxn[x60Sneg,iL0N0Spos+iL0N2Sneg,1],
rxn[x61Spos,iL0N1Sneg+iL0N2Spos,1],
rxn[x61Sneg,iL0N1Spos+iL0N2Sneg,1],
rxn[x62Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x62Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
rxn[x63Spos,iL0N0Sneg+iL0N1Spos+iL0N2Spos+iL0N3Spos,1],
rxn[x63Sneg,iL0N0Spos+iL0N1Sneg+iL0N2Sneg+iL0N3Sneg,1],
conc[iL0N0Sneg,0.0491438442],
conc[iL0N1Sneg,0.1034315803],
conc[iL0N2Sneg,0.0240748579],
conc[iL0N3Sneg,0.1202642315],
rxn[iL0N0Spos,mL0N0+hL0N0Spos,1],
rxn[mL0N0+iL0N0Sneg,hL0N0Sneg,1],
rxn[iL0N1Spos,mL0N1+hL0N1Spos,1],
rxn[mL0N1+iL0N1Sneg,hL0N1Sneg,1],
rxn[iL0N2Spos,mL0N2+hL0N2Spos,1],
rxn[mL0N2+iL0N2Sneg,hL0N2Sneg,1],
rxn[iL0N3Spos,mL0N3+hL0N3Spos,1],
rxn[mL0N3+iL0N3Sneg,hL0N3Sneg,1],
rxn[hL0N0Spos,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N0Sneg,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N1Spos,hL1N1Spos,1],
rxn[hL0N1Sneg,hL1N1Sneg,1],
rxn[hL0N2Spos,hL1N0Sneg+hL1N1Spos,1],
rxn[hL0N2Sneg,hL1N0Spos+hL1N1Sneg,1],
rxn[hL0N3Spos,hL1N0Spos+hL1N1Spos,1],
rxn[hL0N3Sneg,hL1N0Sneg+hL1N1Sneg,1],
conc[hL1N0Sneg,0.0036730287],
conc[hL1N1Sneg,0.1354081898]
}

MNISTReduced[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_,x4SposInitial_,x4SnegInitial_,x5SposInitial_,x5SnegInitial_,x6SposInitial_,x6SnegInitial_,x7SposInitial_,x7SnegInitial_,x8SposInitial_,x8SnegInitial_,x9SposInitial_,x9SnegInitial_,x10SposInitial_,x10SnegInitial_,x11SposInitial_,x11SnegInitial_,x12SposInitial_,x12SnegInitial_,x13SposInitial_,x13SnegInitial_,x14SposInitial_,x14SnegInitial_,x15SposInitial_,x15SnegInitial_,x16SposInitial_,x16SnegInitial_,x17SposInitial_,x17SnegInitial_,x18SposInitial_,x18SnegInitial_,x19SposInitial_,x19SnegInitial_,x20SposInitial_,x20SnegInitial_,x21SposInitial_,x21SnegInitial_,x22SposInitial_,x22SnegInitial_,x23SposInitial_,x23SnegInitial_,x24SposInitial_,x24SnegInitial_,x25SposInitial_,x25SnegInitial_,x26SposInitial_,x26SnegInitial_,x27SposInitial_,x27SnegInitial_,x28SposInitial_,x28SnegInitial_,x29SposInitial_,x29SnegInitial_,x30SposInitial_,x30SnegInitial_,x31SposInitial_,x31SnegInitial_,x32SposInitial_,x32SnegInitial_,x33SposInitial_,x33SnegInitial_,x34SposInitial_,x34SnegInitial_,x35SposInitial_,x35SnegInitial_,x36SposInitial_,x36SnegInitial_,x37SposInitial_,x37SnegInitial_,x38SposInitial_,x38SnegInitial_,x39SposInitial_,x39SnegInitial_,x40SposInitial_,x40SnegInitial_,x41SposInitial_,x41SnegInitial_,x42SposInitial_,x42SnegInitial_,x43SposInitial_,x43SnegInitial_,x44SposInitial_,x44SnegInitial_,x45SposInitial_,x45SnegInitial_,x46SposInitial_,x46SnegInitial_,x47SposInitial_,x47SnegInitial_,x48SposInitial_,x48SnegInitial_,x49SposInitial_,x49SnegInitial_,x50SposInitial_,x50SnegInitial_,x51SposInitial_,x51SnegInitial_,x52SposInitial_,x52SnegInitial_,x53SposInitial_,x53SnegInitial_,x54SposInitial_,x54SnegInitial_,x55SposInitial_,x55SnegInitial_,x56SposInitial_,x56SnegInitial_,x57SposInitial_,x57SnegInitial_,x58SposInitial_,x58SnegInitial_,x59SposInitial_,x59SnegInitial_,x60SposInitial_,x60SnegInitial_,x61SposInitial_,x61SnegInitial_,x62SposInitial_,x62SnegInitial_,x63SposInitial_,x63SnegInitial_] =
  reduceFFNCCrn[MNIST[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial,x10SposInitial,x10SnegInitial,x11SposInitial,x11SnegInitial,x12SposInitial,x12SnegInitial,x13SposInitial,x13SnegInitial,x14SposInitial,x14SnegInitial,x15SposInitial,x15SnegInitial,x16SposInitial,x16SnegInitial,x17SposInitial,x17SnegInitial,x18SposInitial,x18SnegInitial,x19SposInitial,x19SnegInitial,x20SposInitial,x20SnegInitial,x21SposInitial,x21SnegInitial,x22SposInitial,x22SnegInitial,x23SposInitial,x23SnegInitial,x24SposInitial,x24SnegInitial,x25SposInitial,x25SnegInitial,x26SposInitial,x26SnegInitial,x27SposInitial,x27SnegInitial,x28SposInitial,x28SnegInitial,x29SposInitial,x29SnegInitial,x30SposInitial,x30SnegInitial,x31SposInitial,x31SnegInitial,x32SposInitial,x32SnegInitial,x33SposInitial,x33SnegInitial,x34SposInitial,x34SnegInitial,x35SposInitial,x35SnegInitial,x36SposInitial,x36SnegInitial,x37SposInitial,x37SnegInitial,x38SposInitial,x38SnegInitial,x39SposInitial,x39SnegInitial,x40SposInitial,x40SnegInitial,x41SposInitial,x41SnegInitial,x42SposInitial,x42SnegInitial,x43SposInitial,x43SnegInitial,x44SposInitial,x44SnegInitial,x45SposInitial,x45SnegInitial,x46SposInitial,x46SnegInitial,x47SposInitial,x47SnegInitial,x48SposInitial,x48SnegInitial,x49SposInitial,x49SnegInitial,x50SposInitial,x50SnegInitial,x51SposInitial,x51SnegInitial,x52SposInitial,x52SnegInitial,x53SposInitial,x53SnegInitial,x54SposInitial,x54SnegInitial,x55SposInitial,x55SnegInitial,x56SposInitial,x56SnegInitial,x57SposInitial,x57SnegInitial,x58SposInitial,x58SnegInitial,x59SposInitial,x59SnegInitial,x60SposInitial,x60SnegInitial,x61SposInitial,x61SnegInitial,x62SposInitial,x62SnegInitial,x63SposInitial,x63SnegInitial], keepInputLayer=True, inputPrefix="x"];

cnt = Count[MNIST[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial,x10SposInitial,x10SnegInitial,x11SposInitial,x11SnegInitial,x12SposInitial,x12SnegInitial,x13SposInitial,x13SnegInitial,x14SposInitial,x14SnegInitial,x15SposInitial,x15SnegInitial,x16SposInitial,x16SnegInitial,x17SposInitial,x17SnegInitial,x18SposInitial,x18SnegInitial,x19SposInitial,x19SnegInitial,x20SposInitial,x20SnegInitial,x21SposInitial,x21SnegInitial,x22SposInitial,x22SnegInitial,x23SposInitial,x23SnegInitial,x24SposInitial,x24SnegInitial,x25SposInitial,x25SnegInitial,x26SposInitial,x26SnegInitial,x27SposInitial,x27SnegInitial,x28SposInitial,x28SnegInitial,x29SposInitial,x29SnegInitial,x30SposInitial,x30SnegInitial,x31SposInitial,x31SnegInitial,x32SposInitial,x32SnegInitial,x33SposInitial,x33SnegInitial,x34SposInitial,x34SnegInitial,x35SposInitial,x35SnegInitial,x36SposInitial,x36SnegInitial,x37SposInitial,x37SnegInitial,x38SposInitial,x38SnegInitial,x39SposInitial,x39SnegInitial,x40SposInitial,x40SnegInitial,x41SposInitial,x41SnegInitial,x42SposInitial,x42SnegInitial,x43SposInitial,x43SnegInitial,x44SposInitial,x44SnegInitial,x45SposInitial,x45SnegInitial,x46SposInitial,x46SnegInitial,x47SposInitial,x47SnegInitial,x48SposInitial,x48SnegInitial,x49SposInitial,x49SnegInitial,x50SposInitial,x50SnegInitial,x51SposInitial,x51SnegInitial,x52SposInitial,x52SnegInitial,x53SposInitial,x53SnegInitial,x54SposInitial,x54SnegInitial,x55SposInitial,x55SnegInitial,x56SposInitial,x56SnegInitial,x57SposInitial,x57SnegInitial,x58SposInitial,x58SnegInitial,x59SposInitial,x59SnegInitial,x60SposInitial,x60SnegInitial,x61SposInitial,x61SnegInitial,x62SposInitial,x62SnegInitial,x63SposInitial,x63SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];
cntReduced = Count[MNISTReduced[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial,x4SposInitial,x4SnegInitial,x5SposInitial,x5SnegInitial,x6SposInitial,x6SnegInitial,x7SposInitial,x7SnegInitial,x8SposInitial,x8SnegInitial,x9SposInitial,x9SnegInitial,x10SposInitial,x10SnegInitial,x11SposInitial,x11SnegInitial,x12SposInitial,x12SnegInitial,x13SposInitial,x13SnegInitial,x14SposInitial,x14SnegInitial,x15SposInitial,x15SnegInitial,x16SposInitial,x16SnegInitial,x17SposInitial,x17SnegInitial,x18SposInitial,x18SnegInitial,x19SposInitial,x19SnegInitial,x20SposInitial,x20SnegInitial,x21SposInitial,x21SnegInitial,x22SposInitial,x22SnegInitial,x23SposInitial,x23SnegInitial,x24SposInitial,x24SnegInitial,x25SposInitial,x25SnegInitial,x26SposInitial,x26SnegInitial,x27SposInitial,x27SnegInitial,x28SposInitial,x28SnegInitial,x29SposInitial,x29SnegInitial,x30SposInitial,x30SnegInitial,x31SposInitial,x31SnegInitial,x32SposInitial,x32SnegInitial,x33SposInitial,x33SnegInitial,x34SposInitial,x34SnegInitial,x35SposInitial,x35SnegInitial,x36SposInitial,x36SnegInitial,x37SposInitial,x37SnegInitial,x38SposInitial,x38SnegInitial,x39SposInitial,x39SnegInitial,x40SposInitial,x40SnegInitial,x41SposInitial,x41SnegInitial,x42SposInitial,x42SnegInitial,x43SposInitial,x43SnegInitial,x44SposInitial,x44SnegInitial,x45SposInitial,x45SnegInitial,x46SposInitial,x46SnegInitial,x47SposInitial,x47SnegInitial,x48SposInitial,x48SnegInitial,x49SposInitial,x49SnegInitial,x50SposInitial,x50SnegInitial,x51SposInitial,x51SnegInitial,x52SposInitial,x52SnegInitial,x53SposInitial,x53SnegInitial,x54SposInitial,x54SnegInitial,x55SposInitial,x55SnegInitial,x56SposInitial,x56SnegInitial,x57SposInitial,x57SnegInitial,x58SposInitial,x58SnegInitial,x59SposInitial,x59SnegInitial,x60SposInitial,x60SnegInitial,x61SposInitial,x61SnegInitial,x62SposInitial,x62SnegInitial,x63SposInitial,x63SnegInitial], rxn[___]];
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
rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.245187,0,0.089398,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.115984,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.449554,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.378822,0,0.362906,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.270866,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.445371,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-1.328425488363836, 1.338289288926561];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.800291,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.115954,0.474421,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.182318,0.523861,0,0.509857,0,0.113759,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.661473,0,0.088318,0,0,0.401917,0.518140,0,0,0.002714,0,0.000828,0,0.000064,0,0.044627,0.696128,0,0,0.396469,0,0.536435,0.542597,0,0.005161,0,0,0.001187,0,0.000276,0,0.064708,0.735422,0,0.213513,0,0.463444,0,0.658498,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.011103,0,0.514488,0,0.302510,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 2.845110998252044,-2.984192216739122];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.519036,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.261854,0,0.091114,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.424317,0,0,0.354868,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.580191,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.206027,0.063565,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.472581,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.489979,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.409129003206999, 2.677193599660658];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.978157,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.041316,0.512232,0,0.599421,0,0.537072,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.191947,0,0,0.383737,0,0.459261,0,0.112957,0.181636,0,0,0.004560,0,0.000389,0,0.028086,0.598238,0,0,0.317319,0,0.401917,0,0.356615,0.542874,0,0,0.000828,0,0.000064,0,0.044627,0.297108,0,0,0.396469,0,0.541582,0,0.332403,0.529425,0,0,0.001187,0,0.000276,0,0.064708,0.330520,0,0,0.343105,0,0.473076,0.485214,0,0,0.060715,0,0.001367,0,0.000085,0,0.028514,0.479485,0,0.545126,0,0.489029,0,0,0.024030,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.50074828671814 ,-1.639829505205218];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "10.911184,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.168160,0.430058,0,0,0.024202,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.120063,0.503028,0,0,0.347742,0.574788,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.546277,0,0,0.204329,0,0.401917,0.052944,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.702990,0,0,0.396469,0,0.541582,0.008773,0,0.108837,0,0,0.001187,0,0.000276,0,0.064708,0.689343,0,0,0.343105,0,0.473076,0.330067,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.206544,0.240959,0,0,0.010726,0.444107,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.200013051503906,-1.339094269990984];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.515549,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.431529,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.322111,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.473573,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.208418,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.402415,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.361792,0,0,0.253373,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.211089866380006, 3.686752634961866];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.943772,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.589862,0,0.646141,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.514496,0,0.256214,0,0.226278,0,0.513024,0,0.566685,0,0,0.004560,0,0.000389,0.220444,0,0.684022,0,0.160867,0,0,0.401917,0,0.356615,0.894590,0,0,0.000828,0,0.000064,0.702432,0,0.661569,0,0,0.396469,0,0.432268,0.626666,0,0.774768,0,0,0.001187,0,0.000276,0.231615,0,0.737383,0,0.649052,0,0.519081,0,0.618547,0,0,0.042332,0,0.001367,0,0.000085,0,0.028514,0.477279,0,0.486792,0,0.126039,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 5.06128742429019 ,-5.200368642777268];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "14.092916,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.028080,0.543605,0,0.357264,0,0,0.109496,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.366508,0,0.297847,0,0.583612,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.214262,0.035867,0,0,0.401917,0.585297,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.648824,0,0,0.396469,0,0.541582,0.497989,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.737383,0,0,0.319576,0,0.473076,0.531047,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.177034,0,0.541204,0,0.597608,0,0,0.018393,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.959571667709956,-2.098652886197034];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.918819,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.749242,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.208002,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.128269,0,0,0.401917,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.129006,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.233542,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.421382,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-1.477199963223073, 1.338118744735995];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.005215,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.242134,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.207266,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.195142,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.324104,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.506336,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.538049,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.717707442263872, 4.538713174779676];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.064205,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.284781,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.034645,0,0,0.273967,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.140114,0.192936,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.222388,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.288689,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.367216,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.052266245852024, 2.972046668548845];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.079329,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.505548,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.534121,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.212828,0,0.423573,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.393727,0,0,0.068053,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.332385,0,0.010013,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.129194,0,0.107657,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.500540952055023, 3.086497768379964];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.150768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.399702,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.157064,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.417317,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.257570,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.706121629964444, 1.514514951158007];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.908312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.033228,0.720321,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.508420,0,0,0.430585,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.124837,0,0.382397,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.449594,0,0,0.331423,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.518101,0,0,0.041257,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.103618,0.425235,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.033317882209409, 1.613279900348877];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.515631,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.134336,0.372705,0,0,0.169055,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.522390,0,0,0.446026,0.384102,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.289017,0.670916,0,0,0.401917,0.446571,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.057794,0.180002,0,0,0.541582,0.464656,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.006735,0,0.279134,0,0.473076,0.595753,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.213897,0.455665,0,0.584372,0,0,0.091677,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.309995729024244,-2.434503565727712];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.747871,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.038146,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.534366,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.457152,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.389116,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.268660,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.334096,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-1.347788280736539, 1.208707062249461];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.181730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.174777,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.232388,0,0.424841,0,0.227504,0,0.123808,0,0.513989,0,0,0.004560,0,0.000389,0.114561,0,0.107552,0,0,0.317319,0,0.401917,0,0.356615,0.377188,0,0,0.000828,0,0.000064,0.703167,0,0,0.288186,0,0.396469,0,0.541582,0,0.332403,0.618886,0,0,0.001187,0,0.000276,0.121321,0,0.498167,0,0,0.133546,0,0.030428,0.584234,0,0.113305,0,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.426688,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.919375772351036,-1.058456990838114];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.654154,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.744095,0,0.644764,0,0.201533,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.394643,0,0.231949,0,0,0.454849,0.443416,0,0.419871,0,0,0.004560,0,0.000389,0,0.028086,0.684022,0,0,0.310702,0,0.401917,0,0.356615,0.775227,0,0,0.000828,0,0.000064,0,0.044627,0.582157,0,0,0.396469,0,0.541582,0,0.327991,0.840455,0,0,0.001187,0,0.000276,0,0.064708,0.686157,0,0,0.343105,0,0.437781,0.614871,0,0.473109,0,0,0.001367,0,0.000085,0,0.028514,0.500563,0,0.404930,0,0.317216,0,0.280382,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.181630688482532,-3.320711906969611];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "13.209335,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.154924,0,0.086854,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.008175,0,0.005445,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.153269,0,0.062789,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.272939,0.225085,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.219576,0.028885,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.084341,0,0,0.374696,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.077145479475349,1.487978002895378];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "5.442573,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.202540,0.032906,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.271131,0,0,0.229133,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.222466,0.497348,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.046914,0,0.216506,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.582630,0,0,0.384105,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.452234,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.69221731103471, 3.32768398793485];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.381178,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.696301,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.463812,0,0,0.309996,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.504984,0,0.538524,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.373139,0,0.418222,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.318415,0,0.515405,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.522117,0,0.047784,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.129151172577727, 1.351515150464271];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.131454,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.035433,0,0.131395,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.446165,0,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.069525,0.178230,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.385134,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.018101,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.009255,0,0.107609,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.308638839195282, 2.595085772593282];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.950826,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.231529,0,0,0.015379,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.202906,0.492979,0,0.007651,0,0.269396,0,0,0.056600,0,0.004560,0,0.000389,0,0.028086,0.662699,0,0,0.276633,0,0.401917,0,0.356615,0.662727,0,0,0.000828,0,0.000064,0,0.044627,0.644902,0,0,0.396469,0,0.541582,0,0.332403,0.629425,0,0,0.001187,0,0.000276,0,0.064708,0.696206,0,0,0.343105,0,0.473076,0.126390,0,0.001050,0,0,0.001367,0,0.000085,0,0.028514,0,0.074437,0.464488,0,0.509127,0,0.186019,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.225258017653237,-1.364339236140315];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.660333,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.510357,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.314023,0,0,0.298986,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.243790,0.404946,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.581718,0,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.152568,0.253709,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.224927,0.398802,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.702935299765203, 2.806426612782915];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.475370,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.047198,0.543605,0,0,0.126560,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.022823,0.700526,0,0,0.462414,0.168680,0,0,0.260015,0,0.099247,0,0.004560,0,0.000389,0.778532,0,0,0.273576,0,0.317319,0,0.401672,0.593385,0,0,0.083596,0,0.000828,0,0.000064,0.818118,0,0,0.288186,0,0.396469,0,0.541582,0,0.166962,0.652464,0,0,0.001187,0,0.000276,0.628674,0,0,0.254774,0,0.343105,0,0.473076,0,0.269933,0.691491,0,0,0.001367,0,0.000085,0,0.028514,0.044191,0,0.129194,0,0.293686,0,0.170088,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[1.787437118781892,1.655701957745059];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "11.398776,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.600891,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.520151,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.223202,0.472593,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.150590,0,0.137830,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.203954,0,0.206336,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.026988,0,0.215010,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.88780552722028 , 4.557585890312455];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.191213,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.152429,0,0.648195,0,0,0.126408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.208788,0.286361,0,0,0.459261,0.559837,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.002650,0,0,0.205065,0,0.401917,0.397551,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.089265,0,0,0.396469,0,0.541582,0.405587,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.122677,0,0,0.293840,0,0.441458,0.427126,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.212427,0.521106,0,0.528980,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.152954178625355,-1.292035397112433];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.280811,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.137099,0,0.106555,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.527013,0,0,0.338692,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.065114,0.570632,0,0,0.354164,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.384414,0,0.214300,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.201098,0.644640,0,0,0.451017,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.013652,0.394635,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.072609486202509, 3.58748805793073 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.804826,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.468048,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.378239,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.435583,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.363380,0,0.283494,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.428954,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.406645,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.408638851861269, 1.33168001703034 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.382956,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.335517,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.286116,0,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.178349,0,0.335741,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.151141,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.315895,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.399568,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.416415712725255, 1.277334494238177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.736180,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.613146,0,0,0.155820,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.380724,0,0,0.459261,0.530181,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.216958,0.029739,0,0,0.401917,0,0.006615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.398579,0,0,0.396469,0,0.541582,0.017597,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.499638,0,0,0.335017,0,0.473076,0.609234,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.171250,0.449538,0,0.453980,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.592414996349821,-1.731496214836899];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.424736,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.180971,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.003379,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.390730,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.312339,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.319817,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.111333,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.119178103427202, 5.80170360477712 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.260084,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.099911,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.216876,0.500332,0,0,0.248232,0.198808,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.159115,0,0.212417,0,0.401917,0.389708,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.215245,0,0,0.396469,0,0.541582,0.417597,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.066049,0,0.233546,0,0.442928,0.660459,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.210512,0.522363,0,0,0.022560,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-0.003673028725123,-0.135408189761956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.508334,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.019630,0,0.164795,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.121237,0,0.106320,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.168055,0.076759,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.325880,0,0.329082,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.128399,0.242366,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.424568,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-2.462739265452955, 2.708632519591291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.323640,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.163324,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.151908,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.279858,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.528347,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.286487,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.092129,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-0.474503948282335, 0.671867830801447];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.517136,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.549176,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.521621,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.403965,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.369263,0.079006,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.278399,0.149719,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.415659,0.162068,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.303001584629328, 4.27498780461099 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.679864,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.324176,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.390739,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.448083,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.169447,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.043836,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.151039,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.083148682634146, 4.573517176230214];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.468835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.700468,0,0.260940,0,0.652023,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.631898,0,0,0.368296,0,0.459261,0,0.267368,0.729185,0,0,0.004560,0,0.000389,0.113091,0,0.551424,0,0,0.317319,0,0.401917,0,0.356615,0.916404,0,0,0.000828,0,0.000064,0.705373,0,0,0.288186,0,0.396469,0,0.541582,0,0.238285,0.702955,0,0,0.001187,0,0.000276,0.685292,0,0.027579,0,0,0.343105,0,0.144399,0.668302,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.629485,0,0.549047,0,0.026774,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.661042484217892,-3.80012370270497 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "13.110508,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.121836,0,0.208177,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.216876,0.526802,0,0.529955,0,0.567926,0,0.674773,0,0,0.004560,0,0.000389,0,0.028086,0.129611,0,0.222632,0,0,0.401917,0,0.356615,0.827433,0,0,0.000828,0,0.000064,0,0.044627,0.702990,0,0,0.396469,0,0.541582,0,0.313285,0.808347,0,0,0.001187,0,0.000276,0,0.064708,0.705030,0,0,0.343105,0,0.426752,0.644528,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.753750,0,0.507381,0,0.447117,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.162022719854126,-3.301103938341204];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "12.126181,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.334046,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.062414,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.277614,0,0.401917,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.029741,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.227170,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.146510,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.163229382526014, 1.840004986274679];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.213265,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.247148,0,0.058026,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.443680,0,0.339985,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.132025,0.589259,0,0,0.232351,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.081518,0.410869,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.645130,0,0,0.278223,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.499783,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.152266280869753, 3.748027029354833];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.844645,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.000139,0.088948,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.506949,0,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.232681,0,0,0.276182,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.349594,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.515160,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.013912,0,0.052216,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.148834929480017, 1.442634913046621];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.605761,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.245677,0.763543,0,0.249676,0,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.318786,0.306425,0,0,0.408545,0.526489,0,0,0.004560,0,0.000389,0,0.028086,0,0.234605,0.420426,0,0,0.401917,0,0.356615,0.487727,0,0,0.000828,0,0.000064,0,0.044627,0.675294,0,0,0.396469,0,0.541582,0.587695,0,0,0.094104,0,0.001187,0,0.000276,0.496321,0,0,0.176098,0,0.343105,0.475699,0,0,0.314786,0,0.063656,0,0.001367,0,0.000085,0.513397,0,0.762083,0,0.045371,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.110552274385104,-0.249633492872182];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "11.416351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.043522,0.355370,0,0.267558,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.584839,0,0.479498,0,0,0.270291,0.530671,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.098973,0,0,0.144280,0,0.401917,0.251473,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.623334,0,0,0.396469,0,0.541582,0.346273,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.407971,0,0,0.343105,0,0.456899,0.654577,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.730956,0,0.355665,0,0.508637,0,0,0.079912,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.546826655739556,-1.685907874226634];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.738136,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.165954,0,0.029501,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.091825,0.373092,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.370916,0,0.376759,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.436865,0,0.143712,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.490228,0,0.212219,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.282136,0,0.065010,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.468678311301634, 2.507576197006025];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.803811,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.040994,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.265144,0,0.148251,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.576024,0,0,0.354409,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.125590,0,0,0.391582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.621111,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.207770,0.447577,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.457837383191917, 2.640250293660601];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.943893,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.216823,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.215144,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.157799,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.231288,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.009277,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.336578,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.028981989393742, 2.598517264147066];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.108424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.249109,0,0.102879,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.136082,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.441465,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.356763,0,0.400406,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.280424,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.413998,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.369847028445337, 1.395397154736956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.823330,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.131105,0,0.651872,0,0.022121,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.073984,0.423861,0,0,0.430340,0.586553,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.628140,0,0,0.249427,0,0.401917,0.638483,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.707892,0,0,0.396469,0,0.430798,0.615881,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.722677,0,0,0.343105,0.486973,0,0,0.171649,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.135858,0,0.544145,0,0,0.009255,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.057120901295552,-2.19620211978263 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.654337,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0,0.038908,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.315389,0,0.397515,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.381416,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.365586,0,0.523935,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.297091,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.098502,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.894111803661916, 2.228485433507402];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.770473,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.164060,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.489513,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.510828,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.186910,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.590719,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.099927,0,0.157080,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-2.547053015719984, 2.789760036516626];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.129158,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.018471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.218435,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.204946,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.095428,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.320797,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.443196,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.483148673693449, 4.835772118438743];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.242787,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.067508,0,0.679879,0,0.554568,0,0,0.121261,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.756163,0,0,0.441090,0,0.397496,0.583122,0,0,0.099247,0,0.004560,0,0.000389,0.371914,0,0,0.011076,0,0.317319,0,0.401917,0,0.298527,0.860031,0,0,0.000828,0,0.000064,0.355373,0,0.114755,0,0,0.396469,0,0.541582,0,0.332403,0.762759,0,0,0.001187,0,0.000276,0,0.064708,0.728314,0,0,0.176929,0,0.473076,0,0.331698,0.928501,0,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.126253,0,0.481921,0,0.675970,0,0.042125,0,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.708346294933567,-0.847427513420645];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "12.923145,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.126248,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.161116,0,0,0.417349,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.055475,0,0,0.401917,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.354557,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0,0.211311,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.387314,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.743866651248071, 0.604785432760993];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.534993,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.026693,0,0.550891,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.359857,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.252614,0.403965,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.177351,0.264300,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.007076,0.350454,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.415659,0.461333,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-3.248589849243671, 4.835036898870729];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.498201,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.432264,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.427013,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.237908,0.579701,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.324410,0.426065,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.263693,0.413199,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.387804,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.636334853390963, 5.965919154603742];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.376678,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.050771,0,0,0.155796,0.272111,0,0.422092,0,0,0.099247,0,0.004560,0,0.000389,0.075591,0,0.013189,0,0,0.317319,0,0.401917,0,0.356615,0.759541,0,0,0.000828,0,0.000064,0.805863,0,0,0.288186,0,0.396469,0,0.541582,0,0.313285,0.870111,0,0,0.001187,0,0.000276,0.786272,0,0,0.254774,0,0.343105,0.146287,0,0.588400,0,0,0.021009,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.271091428348432,-0.305590359063649];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "10.429457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.508311,0,0.339617,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.101680,0.489547,0,0,0.237693,0.054445,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.662209,0,0,0.309476,0,0.401917,0.639463,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.459608,0,0,0.396469,0,0.541582,0.660489,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.706010,0,0,0.343105,0,0.079693,0.664381,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.112427,0.513508,0,0.490255,0,0,0.071089,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.287022801810513,-2.426104020297591];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.698948,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.036676,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.196621,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.253965,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.114300,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.251189,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.496627,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.497119220117838, 4.777193627227805];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.290894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.640842,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.484611,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.464750,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.175145,0.107683,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.452728,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.477479,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.187805475811273, 1.893199565327269];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.421850,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.091823,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.276033,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.333377,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.098935,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.262219,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.386088,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.690256441156179, 4.476458336134217];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.389989,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.271413,0,0.035967,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.540739,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.059151,0,0.083622,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.306473,0,0,0.353347,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.516699,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.470861,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.545092119764421, 1.406010901277343];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.763156,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.175223,0,0.170499,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.201925,0.533665,0,0,0.241614,0.381896,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.219899,0,0.069770,0,0.401917,0.058091,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.382892,0,0,0.396469,0,0.541582,0,0.191227,0,0.090428,0,0.001187,0,0.000276,0,0.064708,0.694245,0,0,0.343105,0,0.473076,0.366832,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.109632,0,0.118655,0,0.421627,0,0.053421,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.421826760047684,-1.560907978534762];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.459560,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.590842,0,0,0.125673,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.064180,0.526067,0,0,0.284261,0.544887,0,0.072077,0,0,0.004560,0,0.000389,0,0.028086,0.678875,0,0,0.283006,0,0.401917,0,0.313233,0.762482,0,0,0.000828,0,0.000064,0.775962,0,0,0.147010,0,0.396469,0,0.541582,0,0.332403,0.898053,0,0,0.001187,0,0.000276,0.850488,0,0,0.206245,0,0.343105,0.003150,0,0.621979,0,0.549824,0,0,0.001367,0,0.000085,0.079574,0,0.749583,0,0.499537,0,0.210598,0,0.123029,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 4.047561933049569,-4.186643151536647];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "12.442863,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.070449,0,0,0.205660,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.748810,0,0,0.292806,0.423092,0,0.134838,0,0.487518,0,0,0.004560,0,0.000389,0.283679,0,0,0.077987,0,0.317319,0,0.401917,0,0.356615,0.187727,0,0,0.000828,0,0.000064,0.702432,0,0,0.288186,0,0.396469,0,0.541582,0,0.332403,0.736288,0,0,0.001187,0,0.000276,0.205145,0,0.103314,0,0,0.343105,0,0.409840,0.593792,0,0,0.001891,0,0.001367,0,0.000085,0,0.028514,0,0.024192,0,0.130364,0,0.149696,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[1.004594031913367,0.872858870876535];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.764194,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.252295,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.048967,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.257761,0.216465,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.264116,0.076800,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.287958,0.251189,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.505696,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.634129041950018, 5.061752500851415];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.045868,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.229272,0,0.538458,0,0.427117,0,0,0.062437,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.551751,0,0,0.310208,0,0.459261,0,0.408545,0.755900,0,0,0.004560,0,0.000389,0,0.028086,0.083042,0,0,0.317319,0,0.401917,0,0.356615,0.031110,0,0.156280,0,0,0.000064,0,0.044627,0,0.005833,0,0.396469,0,0.541582,0,0.332403,0.034817,0,0.125528,0,0,0.000276,0,0.064708,0.594000,0,0,0.343105,0,0.473076,0,0.331698,0.795168,0,0,0.001367,0,0.000085,0,0.028514,0,0.117084,0.446596,0,0.227019,0,0.686754,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.035797351326896,-0.174878569813974];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "10.885704,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.090302,0,0.783556,0,0.643784,0,0.180700,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.762290,0,0.508420,0,0.374072,0,0.580671,0,0,0.011747,0,0.004560,0,0.000389,0,0.028086,0.680101,0,0,0.221976,0,0.401917,0.565934,0,0.750963,0,0,0.000828,0,0.000064,0,0.044627,0.700049,0,0,0.343282,0,0.541582,0.513185,0,0.808347,0,0,0.001187,0,0.000276,0,0.064708,0.736402,0,0.648071,0,0.422758,0,0.659479,0,0.574579,0,0,0.001367,0,0.000085,0,0.028514,0,0.146250,0.400028,0,0.596627,0,0.414941,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.192905108207474,-3.331986326694552];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "14.409365,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.055660,0.601627,0,0.340014,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.128641,0.340037,0,0,0.459261,0.450524,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.553140,0,0,0.145996,0,0.401917,0.435542,0,0,0.071096,0,0.000828,0,0.000064,0,0.044627,0.703971,0,0,0.349410,0,0.541582,0.659754,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.737383,0,0,0.144085,0.210993,0,0.383988,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.094290,0.541204,0,0,0.206314,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.036042249524603,-2.175123468011681];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.324663,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.587411,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.036467,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.395142,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.239790,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.402415,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.402510,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.569913374404222, 5.062242763955854];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.005249,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.392870,0,0,0.269942,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.217367,0,0.031286,0,0.133526,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.012546,0,0.267810,0,0.190397,0.105640,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.312304,0,0,0.396469,0,0.541582,0.066862,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.056010,0,0,0.343105,0,0.473076,0.346979,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.138407,0,0.177668,0,0.394549,0.441656,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.003673028725123, 0.242458514655671];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.634569,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.064483,0.681529,0,0.215749,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.195553,0.537586,0,0,0.459261,0,0.408545,0.650753,0,0,0.004560,0,0.000389,0,0.028086,0.691866,0,0,0.270506,0,0.401917,0,0.215439,0.666404,0,0,0.000828,0,0.000064,0,0.044627,0.633628,0,0,0.396469,0,0.541582,0.574215,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.667039,0,0,0.343105,0.182562,0,0.388155,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.286593,0,0.549047,0,0.309127,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.906875747019062,-4.04595696550614 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "11.000101,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.587411,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.485837,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.543181,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.164116,0.275084,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.160752,0.413199,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.178980,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.884619252900392, 5.915183800180219];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.010501,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.513391,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.338288,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.350044,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.312339,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.328150,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.296872,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.272364305476458, 5.042634802778028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.080599,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.536921,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.511327,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.530436,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.170183,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.281340,0,0.047340,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.035726,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.94858980752042 , 4.344595628698133];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.711682,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.239617,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.349562,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.252614,0.221612,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.331763,0,0.071729,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.278399,0,0.133370,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.390659,0,0.079843,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.694178024451048, 3.247781979759   ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.782836,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.315352,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.529955,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.586318,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.100880,0.373369,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.377905,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.320157,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.816236825804026, 6.525968092401288];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.148981,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.290352,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.365283,0,0.507651,0,0.583122,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.641259,0,0,0.401917,0.558091,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.200931,0,0.041812,0,0.541582,0.622499,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.064578,0,0.177419,0,0.042928,0,0.149590,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.126397,0.545126,0,0.097363,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.089407559032274,-0.408033053575994];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.726858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.264862,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.503484,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.343426,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.026816,0,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.532385,0,0,0.278223,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.424783,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.01788628001127 , 1.878805061524192];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.061097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.544340,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.362341,0,0.148827,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.092239,0,0.387054,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.430967,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.523003,0,0,0.249345,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.321137,0,0,0.045109,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.853915634494159, 2.693681402254541];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.934992,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.205945,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.459261,0.411798,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.334112,0,0,0.307351,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.047649,0,0,0.418053,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.181245,0.113268,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.675073,0,0,0.450953,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-3.455876509588096, 3.624073749530752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.115376,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.269029,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.398582,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.062789,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.213641,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0,0.093664,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.059255,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.733148673693449, 2.816899674852155];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.038474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.380793,0,0,0.059496,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.292243,0,0,0.459261,0.085573,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.287546,0.207190,0,0,0.401917,0.040444,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.001912,0,0.381027,0,0.541582,0.538185,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.137383,0,0,0.343105,0,0.439987,0,0.014541,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.163897,0.094145,0,0.386088,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.059081586772213,-0.198162805259291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.540735,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.830743,0,0.580370,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.349545,0,0,0.462414,0.366964,0,0,0.190653,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.128723,0,0.317319,0,0.401917,0,0.123527,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.051421,0,0.396469,0,0.541582,0,0.332403,0.530896,0,0,0.001187,0,0.000276,0,0.064708,0.518755,0,0,0.343105,0,0.473076,0,0.133168,0.195168,0,0,0.001367,0,0.000085,0,0.028514,0,0.220515,0.385812,0,0.325549,0,0.620578,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-0.003673028725123,-0.135408189761956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "9.845747,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.060940,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.367700,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.339995,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.097464,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.168840,0,0.451017,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.255665,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.669178011040003, 1.683811313380264];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.283627,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.286676,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.301278,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.282025,0.419897,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.317057,0,0.198935,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.307811,0,0.297340,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.424482,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.589521171669275, 3.170821177680753];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.663718,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.173752,0,0,0.034648,0,0.176408,0.432029,0,0,0.001175,0,0.000642,0,0.027970,0.759839,0,0,0.402119,0,0.459261,0,0.408545,0.788008,0,0,0.004560,0,0.000389,0.188826,0,0,0.133870,0,0.317319,0,0.401917,0,0.356615,0.182090,0,0,0.000828,0,0.000064,0.534785,0,0,0.263186,0,0.396469,0,0.541582,0,0.332403,0.171582,0,0,0.001187,0,0.000276,0,0.057355,0.706500,0,0,0.343105,0,0.473076,0,0.331698,0.890511,0,0,0.001367,0,0.000085,0,0.028514,0,0.088652,0.301988,0,0.326039,0,0.639940,0,0,0.008610,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.64388556009681 ,-0.782966778583888];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "11.837839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.374911,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.380935,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.162789,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.275880,0,0.425406,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.222517,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.375953,0,0.379843,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.400239280354116, 1.930691419291933];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.822297,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.374327,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.180340,0.027975,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.527005,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.337110,0,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.253464,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.328995,0,0,0.450953,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-4.023523635160062, 4.612309066701372];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "8.143806,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.232264,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.390739,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.406171,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.279251,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.209282,0,0.114987,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.353949,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.555942842880995, 2.873762431462072];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.528649,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.064483,0.431529,0,0,0.129349,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.023249,0.537586,0,0.540739,0,0.403220,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.691866,0,0,0.317319,0,0.401917,0.596326,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.414755,0,0,0.396469,0,0.541582,0.044068,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.260667,0,0,0.343105,0,0.473076,0.418302,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.598357,0,0.549047,0,0.605451,0,0.731607,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.406630637818585,-1.545711856305663];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "10.253635,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.007999,0,0,0.083026,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.368296,0.080445,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.223202,0.051759,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.300145,0.234398,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.325458,0.426924,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.526039,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.399079984198124, 4.809301505704187];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.873491,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.671725,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.415004,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.448083,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.056763,0.154741,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.295130,0,0,0.292928,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.068900,0,0.069421,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.137560418019802, 3.162732894439958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.223683,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.453833,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.267945,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.094407,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.046729,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.252905,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.394666,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.398344690124304, 3.722291710694097];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.451999,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.031478,0,0.368360,0,0.616087,0,0.053984,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.416702,0,0,0.425649,0,0.459261,0,0.130604,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.610983,0,0,0.317319,0,0.401917,0,0.356615,0.431110,0,0,0.000828,0,0.000064,0,0.044627,0.370637,0,0,0.396469,0,0.541582,0,0.332403,0.696827,0,0,0.001187,0,0.000276,0,0.064708,0.597922,0,0,0.343105,0,0.473076,0.529577,0,0,0.022479,0,0.001367,0,0.000085,0,0.028514,0,0.197721,0.445861,0,0.568931,0,0.115676,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.761287555792937,-0.900368774280016];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "10.698936,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.036118,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.044555,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.025446,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.121729,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.133542,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.407166,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.246629022205979, 3.583811408747696];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.710073,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.125156,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.503484,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.122103,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.038380,0,0.395259,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.451013,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.264243,0,0,0.106314,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.903915720920894, 1.764834502433816];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.230015,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.041772,0,0.124487,0,0.381529,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.765231,0,0.529743,0,0.532896,0,0.583612,0,0.063744,0,0,0.004560,0,0.000389,0.239561,0,0.684022,0,0,0.226878,0,0.390888,0.370100,0,0.784786,0,0,0.000828,0,0.000064,0.223020,0,0.594412,0,0,0.396469,0,0.541582,0.180832,0,0.898053,0,0,0.001187,0,0.000276,0.202939,0,0.720961,0,0,0.219576,0.200944,0,0.660459,0,0.603256,0,0,0.001367,0,0.000085,0,0.028514,0.714289,0,0.541204,0,0.597608,0,0.207588,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.063738430613289,-3.202819649100367];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "13.608782,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.148307,0.215597,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.511116,0,0,0.459261,0.482877,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.089262,0,0.102614,0,0.401917,0.279414,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.488285,0,0,0.332743,0,0.541582,0.253871,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.571206,0,0,0.251438,0,0.435576,0.340361,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.179339,0.463998,0,0.070157,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.264718892747174,-1.403800111234252];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "7.874085,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
];

rsys=MNIST[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.203587,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.059227,0,0.459261,0,0.216388,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.380965,0,0,0.401917,0.015934,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.061175,0,0.541582,0,0.124805,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.122517,0,0.439252,0,0.273119,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.124537,0,0.353735,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[0.141848918242542,0.010113757205709];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNIST_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNIST_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNIST_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNIST_convergence_times.csv", ","];
  ];
  WriteString["MNIST_convergence_times.csv", "6.319996,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNIST_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNIST_convergence_times.csv", ""];
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
rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.245187,0,0.089398,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.115984,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.449554,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.378822,0,0.362906,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.270866,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.445371,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-1.328425488363836, 1.338289288926561];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p0=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p0.svg"}], p0];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.800291,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.115954,0.474421,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.182318,0.523861,0,0.509857,0,0.113759,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.661473,0,0.088318,0,0,0.401917,0.518140,0,0,0.002714,0,0.000828,0,0.000064,0,0.044627,0.696128,0,0,0.396469,0,0.536435,0.542597,0,0.005161,0,0,0.001187,0,0.000276,0,0.064708,0.735422,0,0.213513,0,0.463444,0,0.658498,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.011103,0,0.514488,0,0.302510,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["1: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 2.845110998252044,-2.984192216739122];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p1=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p1.svg"}], p1];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.519036,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.261854,0,0.091114,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.424317,0,0,0.354868,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.580191,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.206027,0.063565,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.472581,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.489979,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["2: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.409129003206999, 2.677193599660658];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p2=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p2.svg"}], p2];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.978157,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.041316,0.512232,0,0.599421,0,0.537072,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.191947,0,0,0.383737,0,0.459261,0,0.112957,0.181636,0,0,0.004560,0,0.000389,0,0.028086,0.598238,0,0,0.317319,0,0.401917,0,0.356615,0.542874,0,0,0.000828,0,0.000064,0,0.044627,0.297108,0,0,0.396469,0,0.541582,0,0.332403,0.529425,0,0,0.001187,0,0.000276,0,0.064708,0.330520,0,0,0.343105,0,0.473076,0.485214,0,0,0.060715,0,0.001367,0,0.000085,0,0.028514,0.479485,0,0.545126,0,0.489029,0,0,0.024030,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.50074828671814 ,-1.639829505205218];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p3=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p3.svg"}], p3];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "10.911184,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.168160,0.430058,0,0,0.024202,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.120063,0.503028,0,0,0.347742,0.574788,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.546277,0,0,0.204329,0,0.401917,0.052944,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.702990,0,0,0.396469,0,0.541582,0.008773,0,0.108837,0,0,0.001187,0,0.000276,0,0.064708,0.689343,0,0,0.343105,0,0.473076,0.330067,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.206544,0.240959,0,0,0.010726,0.444107,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.200013051503906,-1.339094269990984];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p4=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p4.svg"}], p4];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.515549,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.431529,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.322111,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.473573,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.208418,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.402415,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.361792,0,0,0.253373,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["5: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.211089866380006, 3.686752634961866];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p5=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p5.svg"}], p5];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.943772,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.589862,0,0.646141,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.514496,0,0.256214,0,0.226278,0,0.513024,0,0.566685,0,0,0.004560,0,0.000389,0.220444,0,0.684022,0,0.160867,0,0,0.401917,0,0.356615,0.894590,0,0,0.000828,0,0.000064,0.702432,0,0.661569,0,0,0.396469,0,0.432268,0.626666,0,0.774768,0,0,0.001187,0,0.000276,0.231615,0,0.737383,0,0.649052,0,0.519081,0,0.618547,0,0,0.042332,0,0.001367,0,0.000085,0,0.028514,0.477279,0,0.486792,0,0.126039,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 5.06128742429019 ,-5.200368642777268];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p6=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p6.svg"}], p6];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "14.092916,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.028080,0.543605,0,0.357264,0,0,0.109496,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.366508,0,0.297847,0,0.583612,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.214262,0.035867,0,0,0.401917,0.585297,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.648824,0,0,0.396469,0,0.541582,0.497989,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.737383,0,0,0.319576,0,0.473076,0.531047,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.177034,0,0.541204,0,0.597608,0,0,0.018393,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["7: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 1.959571667709956,-2.098652886197034];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p7=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p7.svg"}], p7];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.918819,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.749242,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.208002,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.128269,0,0,0.401917,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.129006,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.233542,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.421382,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-1.477199963223073, 1.338118744735995];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p8=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p8.svg"}], p8];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.005215,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.242134,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.207266,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.195142,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.324104,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.506336,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.538049,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["9: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.717707442263872, 4.538713174779676];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p9=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p9.svg"}], p9];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.064205,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.284781,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.034645,0,0,0.273967,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.140114,0.192936,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.222388,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.288689,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.367216,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["10: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.052266245852024, 2.972046668548845];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p10=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p10.svg"}], p10];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.079329,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.505548,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.534121,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.212828,0,0.423573,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.393727,0,0,0.068053,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.332385,0,0.010013,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.129194,0,0.107657,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["11: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.500540952055023, 3.086497768379964];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p11=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p11.svg"}], p11];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.150768,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.399702,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.157064,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.417317,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.257570,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["12: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.706121629964444, 1.514514951158007];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p12=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p12.svg"}], p12];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.908312,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.033228,0.720321,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.508420,0,0,0.430585,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.124837,0,0.382397,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.449594,0,0,0.331423,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.518101,0,0,0.041257,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.103618,0.425235,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["13: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.033317882209409, 1.613279900348877];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p13=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p13.svg"}], p13];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.515631,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.134336,0.372705,0,0,0.169055,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.522390,0,0,0.446026,0.384102,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.289017,0.670916,0,0,0.401917,0.446571,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.057794,0.180002,0,0,0.541582,0.464656,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.006735,0,0.279134,0,0.473076,0.595753,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.213897,0.455665,0,0.584372,0,0,0.091677,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.309995729024244,-2.434503565727712];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p14=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p14.svg"}], p14];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.747871,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.038146,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.534366,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.457152,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.389116,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.268660,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.334096,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-1.347788280736539, 1.208707062249461];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p15=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p15.svg"}], p15];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.181730,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.174777,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.232388,0,0.424841,0,0.227504,0,0.123808,0,0.513989,0,0,0.004560,0,0.000389,0.114561,0,0.107552,0,0,0.317319,0,0.401917,0,0.356615,0.377188,0,0,0.000828,0,0.000064,0.703167,0,0,0.288186,0,0.396469,0,0.541582,0,0.332403,0.618886,0,0,0.001187,0,0.000276,0.121321,0,0.498167,0,0,0.133546,0,0.030428,0.584234,0,0.113305,0,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.426688,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["16: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 0.919375772351036,-1.058456990838114];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p16=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p16.svg"}], p16];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.654154,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.744095,0,0.644764,0,0.201533,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.394643,0,0.231949,0,0,0.454849,0.443416,0,0.419871,0,0,0.004560,0,0.000389,0,0.028086,0.684022,0,0,0.310702,0,0.401917,0,0.356615,0.775227,0,0,0.000828,0,0.000064,0,0.044627,0.582157,0,0,0.396469,0,0.541582,0,0.327991,0.840455,0,0,0.001187,0,0.000276,0,0.064708,0.686157,0,0,0.343105,0,0.437781,0.614871,0,0.473109,0,0,0.001367,0,0.000085,0,0.028514,0.500563,0,0.404930,0,0.317216,0,0.280382,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 0;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["17: 0 " <> ToString[predicted] <> " 0 " <> warning];
correctResults=List[ 3.181630688482532,-3.320711906969611];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p17=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p17.svg"}], p17];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "13.209335,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.154924,0,0.086854,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.008175,0,0.005445,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.153269,0,0.062789,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.272939,0.225085,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.219576,0.028885,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.084341,0,0,0.374696,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["18: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[0.077145479475349,1.487978002895378];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p18=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p18.svg"}], p18];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "5.442573,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.202540,0.032906,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.271131,0,0,0.229133,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.222466,0.497348,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.046914,0,0.216506,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.582630,0,0,0.384105,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.452234,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["19: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.69221731103471, 3.32768398793485];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p19=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p19.svg"}], p19];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.381178,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.696301,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.463812,0,0,0.309996,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.504984,0,0.538524,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.373139,0,0.418222,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.318415,0,0.515405,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.522117,0,0.047784,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["20: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.129151172577727, 1.351515150464271];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p20=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p20.svg"}], p20];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.131454,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.035433,0,0.131395,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.446165,0,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.069525,0.178230,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.385134,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.018101,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.009255,0,0.107609,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["21: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.308638839195282, 2.595085772593282];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p21=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p21.svg"}], p21];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.950826,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.231529,0,0,0.015379,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.202906,0.492979,0,0.007651,0,0.269396,0,0,0.056600,0,0.004560,0,0.000389,0,0.028086,0.662699,0,0,0.276633,0,0.401917,0,0.356615,0.662727,0,0,0.000828,0,0.000064,0,0.044627,0.644902,0,0,0.396469,0,0.541582,0,0.332403,0.629425,0,0,0.001187,0,0.000276,0,0.064708,0.696206,0,0,0.343105,0,0.473076,0.126390,0,0.001050,0,0,0.001367,0,0.000085,0,0.028514,0,0.074437,0.464488,0,0.509127,0,0.186019,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.225258017653237,-1.364339236140315];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p22=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p22.svg"}], p22];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.660333,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.510357,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.314023,0,0,0.298986,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.243790,0.404946,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.581718,0,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.152568,0.253709,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.224927,0.398802,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["23: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.702935299765203, 2.806426612782915];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p23=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p23.svg"}], p23];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.475370,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.047198,0.543605,0,0,0.126560,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.022823,0.700526,0,0,0.462414,0.168680,0,0,0.260015,0,0.099247,0,0.004560,0,0.000389,0.778532,0,0,0.273576,0,0.317319,0,0.401672,0.593385,0,0,0.083596,0,0.000828,0,0.000064,0.818118,0,0,0.288186,0,0.396469,0,0.541582,0,0.166962,0.652464,0,0,0.001187,0,0.000276,0.628674,0,0,0.254774,0,0.343105,0,0.473076,0,0.269933,0.691491,0,0,0.001367,0,0.000085,0,0.028514,0.044191,0,0.129194,0,0.293686,0,0.170088,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[1.787437118781892,1.655701957745059];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p24=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p24.svg"}], p24];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "11.398776,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.600891,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.520151,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.223202,0.472593,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.150590,0,0.137830,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.203954,0,0.206336,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.026988,0,0.215010,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["25: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.88780552722028 , 4.557585890312455];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p25=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p25.svg"}], p25];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.191213,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.152429,0,0.648195,0,0,0.126408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.208788,0.286361,0,0,0.459261,0.559837,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.002650,0,0,0.205065,0,0.401917,0.397551,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.089265,0,0,0.396469,0,0.541582,0.405587,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.122677,0,0,0.293840,0,0.441458,0.427126,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.212427,0.521106,0,0.528980,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.152954178625355,-1.292035397112433];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p26=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p26.svg"}], p26];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.280811,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.137099,0,0.106555,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.527013,0,0,0.338692,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.065114,0.570632,0,0,0.354164,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.384414,0,0.214300,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.201098,0.644640,0,0,0.451017,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.013652,0.394635,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["27: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.072609486202509, 3.58748805793073 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p27=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p27.svg"}], p27];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.804826,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.468048,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.378239,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.435583,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.363380,0,0.283494,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.428954,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.406645,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["28: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.408638851861269, 1.33168001703034 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p28=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p28.svg"}], p28];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.382956,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.335517,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.286116,0,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.178349,0,0.335741,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.151141,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.315895,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.399568,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["29: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.416415712725255, 1.277334494238177];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p29=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p29.svg"}], p29];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.736180,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.613146,0,0,0.155820,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.380724,0,0,0.459261,0.530181,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.216958,0.029739,0,0,0.401917,0,0.006615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.398579,0,0,0.396469,0,0.541582,0.017597,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.499638,0,0,0.335017,0,0.473076,0.609234,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.171250,0.449538,0,0.453980,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.592414996349821,-1.731496214836899];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p30=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p30.svg"}], p30];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.424736,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.180971,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.003379,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.390730,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.312339,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.319817,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.111333,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["31: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.119178103427202, 5.80170360477712 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p31=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p31.svg"}], p31];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.260084,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.099911,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.216876,0.500332,0,0,0.248232,0.198808,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.159115,0,0.212417,0,0.401917,0.389708,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.215245,0,0,0.396469,0,0.541582,0.417597,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.066049,0,0.233546,0,0.442928,0.660459,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.210512,0.522363,0,0,0.022560,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-0.003673028725123,-0.135408189761956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p32=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p32.svg"}], p32];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.508334,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.019630,0,0.164795,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.121237,0,0.106320,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.168055,0.076759,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.325880,0,0.329082,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.128399,0.242366,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.424568,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-2.462739265452955, 2.708632519591291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p33=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p33.svg"}], p33];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.323640,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.163324,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.151908,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.279858,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.528347,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.286487,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.092129,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-0.474503948282335, 0.671867830801447];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p34=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p34.svg"}], p34];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.517136,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.549176,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.521621,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.403965,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.369263,0.079006,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.278399,0.149719,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.415659,0.162068,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["35: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.303001584629328, 4.27498780461099 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p35=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p35.svg"}], p35];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.679864,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.324176,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.390739,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.448083,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.169447,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.043836,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.151039,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["36: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.083148682634146, 4.573517176230214];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p36=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p36.svg"}], p36];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.468835,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.700468,0,0.260940,0,0.652023,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.631898,0,0,0.368296,0,0.459261,0,0.267368,0.729185,0,0,0.004560,0,0.000389,0.113091,0,0.551424,0,0,0.317319,0,0.401917,0,0.356615,0.916404,0,0,0.000828,0,0.000064,0.705373,0,0,0.288186,0,0.396469,0,0.541582,0,0.238285,0.702955,0,0,0.001187,0,0.000276,0.685292,0,0.027579,0,0,0.343105,0,0.144399,0.668302,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.629485,0,0.549047,0,0.026774,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.661042484217892,-3.80012370270497 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p37=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p37.svg"}], p37];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "13.110508,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.121836,0,0.208177,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.216876,0.526802,0,0.529955,0,0.567926,0,0.674773,0,0,0.004560,0,0.000389,0,0.028086,0.129611,0,0.222632,0,0,0.401917,0,0.356615,0.827433,0,0,0.000828,0,0.000064,0,0.044627,0.702990,0,0,0.396469,0,0.541582,0,0.313285,0.808347,0,0,0.001187,0,0.000276,0,0.064708,0.705030,0,0,0.343105,0,0.426752,0.644528,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.753750,0,0.507381,0,0.447117,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.162022719854126,-3.301103938341204];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p38=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p38.svg"}], p38];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "12.126181,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.334046,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.062414,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.277614,0,0.401917,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.029741,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.227170,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.146510,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["39: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.163229382526014, 1.840004986274679];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p39=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p39.svg"}], p39];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.213265,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.247148,0,0.058026,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.443680,0,0.339985,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.132025,0.589259,0,0,0.232351,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.081518,0.410869,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.645130,0,0,0.278223,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.499783,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["40: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.152266280869753, 3.748027029354833];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p40=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p40.svg"}], p40];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.844645,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.000139,0.088948,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.506949,0,0,0.459261,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.232681,0,0,0.276182,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.349594,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.515160,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.013912,0,0.052216,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["41: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.148834929480017, 1.442634913046621];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p41=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p41.svg"}], p41];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.605761,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.245677,0.763543,0,0.249676,0,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.318786,0.306425,0,0,0.408545,0.526489,0,0,0.004560,0,0.000389,0,0.028086,0,0.234605,0.420426,0,0,0.401917,0,0.356615,0.487727,0,0,0.000828,0,0.000064,0,0.044627,0.675294,0,0,0.396469,0,0.541582,0.587695,0,0,0.094104,0,0.001187,0,0.000276,0.496321,0,0,0.176098,0,0.343105,0.475699,0,0,0.314786,0,0.063656,0,0.001367,0,0.000085,0.513397,0,0.762083,0,0.045371,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.110552274385104,-0.249633492872182];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p42=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p42.svg"}], p42];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "11.416351,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.043522,0.355370,0,0.267558,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.584839,0,0.479498,0,0,0.270291,0.530671,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.098973,0,0,0.144280,0,0.401917,0.251473,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.623334,0,0,0.396469,0,0.541582,0.346273,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.407971,0,0,0.343105,0,0.456899,0.654577,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.730956,0,0.355665,0,0.508637,0,0,0.079912,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.546826655739556,-1.685907874226634];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p43=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p43.svg"}], p43];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.738136,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.165954,0,0.029501,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.091825,0.373092,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.370916,0,0.376759,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.436865,0,0.143712,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.490228,0,0.212219,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.282136,0,0.065010,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["44: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.468678311301634, 2.507576197006025];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p44=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p44.svg"}], p44];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.803811,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.040994,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.265144,0,0.148251,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.576024,0,0,0.354409,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.125590,0,0,0.391582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.621111,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.207770,0.447577,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["45: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.457837383191917, 2.640250293660601];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p45=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p45.svg"}], p45];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.943893,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.216823,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.215144,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.157799,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.231288,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.009277,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.336578,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["46: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.028981989393742, 2.598517264147066];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p46=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p46.svg"}], p46];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.108424,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.249109,0,0.102879,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.136082,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.441465,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.356763,0,0.400406,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.280424,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.413998,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["47: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.369847028445337, 1.395397154736956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p47=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p47.svg"}], p47];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.823330,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.131105,0,0.651872,0,0.022121,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.073984,0.423861,0,0,0.430340,0.586553,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.628140,0,0,0.249427,0,0.401917,0.638483,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.707892,0,0,0.396469,0,0.430798,0.615881,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.722677,0,0,0.343105,0.486973,0,0,0.171649,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.135858,0,0.544145,0,0,0.009255,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.057120901295552,-2.19620211978263 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p48=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p48.svg"}], p48];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.654337,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0,0.038908,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.315389,0,0.397515,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.381416,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.365586,0,0.523935,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.297091,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.098502,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["49: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.894111803661916, 2.228485433507402];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p49=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p49.svg"}], p49];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.770473,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.164060,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.489513,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.510828,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.186910,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.590719,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.099927,0,0.157080,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-2.547053015719984, 2.789760036516626];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p50=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p50.svg"}], p50];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.129158,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.018471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.218435,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.204946,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.095428,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.320797,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.443196,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["51: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.483148673693449, 4.835772118438743];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p51.svg"}], p51];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.242787,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.067508,0,0.679879,0,0.554568,0,0,0.121261,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.756163,0,0,0.441090,0,0.397496,0.583122,0,0,0.099247,0,0.004560,0,0.000389,0.371914,0,0,0.011076,0,0.317319,0,0.401917,0,0.298527,0.860031,0,0,0.000828,0,0.000064,0.355373,0,0.114755,0,0,0.396469,0,0.541582,0,0.332403,0.762759,0,0,0.001187,0,0.000276,0,0.064708,0.728314,0,0,0.176929,0,0.473076,0,0.331698,0.928501,0,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.126253,0,0.481921,0,0.675970,0,0.042125,0,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.708346294933567,-0.847427513420645];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p52=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p52.svg"}], p52];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "12.923145,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.126248,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.161116,0,0,0.417349,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.055475,0,0,0.401917,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.354557,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0,0.211311,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.387314,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["53: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.743866651248071, 0.604785432760993];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p53=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p53.svg"}], p53];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.534993,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.026693,0,0.550891,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.359857,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.252614,0.403965,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.177351,0.264300,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.007076,0.350454,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.415659,0.461333,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["54: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-3.248589849243671, 4.835036898870729];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p54=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p54.svg"}], p54];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.498201,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.432264,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.427013,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.237908,0.579701,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.324410,0.426065,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.263693,0.413199,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.387804,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["55: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.636334853390963, 5.965919154603742];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p55=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p55.svg"}], p55];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.376678,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.050771,0,0,0.155796,0.272111,0,0.422092,0,0,0.099247,0,0.004560,0,0.000389,0.075591,0,0.013189,0,0,0.317319,0,0.401917,0,0.356615,0.759541,0,0,0.000828,0,0.000064,0.805863,0,0,0.288186,0,0.396469,0,0.541582,0,0.313285,0.870111,0,0,0.001187,0,0.000276,0.786272,0,0,0.254774,0,0.343105,0.146287,0,0.588400,0,0,0.021009,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.271091428348432,-0.305590359063649];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p56=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p56.svg"}], p56];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "10.429457,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.508311,0,0.339617,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.101680,0.489547,0,0,0.237693,0.054445,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.662209,0,0,0.309476,0,0.401917,0.639463,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.459608,0,0,0.396469,0,0.541582,0.660489,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.706010,0,0,0.343105,0,0.079693,0.664381,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.112427,0.513508,0,0.490255,0,0,0.071089,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.287022801810513,-2.426104020297591];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p57=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p57.svg"}], p57];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.698948,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.036676,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.196621,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.253965,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.114300,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.251189,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.496627,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["58: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.497119220117838, 4.777193627227805];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p58=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p58.svg"}], p58];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.290894,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.640842,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.484611,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.464750,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.175145,0.107683,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.452728,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.477479,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["59: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.187805475811273, 1.893199565327269];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p59=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p59.svg"}], p59];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.421850,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.091823,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.276033,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.333377,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.098935,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.262219,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.386088,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["60: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.690256441156179, 4.476458336134217];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p60=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p60.svg"}], p60];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.389989,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.271413,0,0.035967,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.540739,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.059151,0,0.083622,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.306473,0,0,0.353347,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.516699,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.470861,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["61: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.545092119764421, 1.406010901277343];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p61=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p61.svg"}], p61];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.763156,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.175223,0,0.170499,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.201925,0.533665,0,0,0.241614,0.381896,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.219899,0,0.069770,0,0.401917,0.058091,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.382892,0,0,0.396469,0,0.541582,0,0.191227,0,0.090428,0,0.001187,0,0.000276,0,0.064708,0.694245,0,0,0.343105,0,0.473076,0.366832,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.109632,0,0.118655,0,0.421627,0,0.053421,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.421826760047684,-1.560907978534762];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p62=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p62.svg"}], p62];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.459560,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.590842,0,0,0.125673,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.064180,0.526067,0,0,0.284261,0.544887,0,0.072077,0,0,0.004560,0,0.000389,0,0.028086,0.678875,0,0,0.283006,0,0.401917,0,0.313233,0.762482,0,0,0.000828,0,0.000064,0.775962,0,0,0.147010,0,0.396469,0,0.541582,0,0.332403,0.898053,0,0,0.001187,0,0.000276,0.850488,0,0,0.206245,0,0.343105,0.003150,0,0.621979,0,0.549824,0,0,0.001367,0,0.000085,0.079574,0,0.749583,0,0.499537,0,0.210598,0,0.123029,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 4.047561933049569,-4.186643151536647];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p63=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p63.svg"}], p63];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "12.442863,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.070449,0,0,0.205660,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.748810,0,0,0.292806,0.423092,0,0.134838,0,0.487518,0,0,0.004560,0,0.000389,0.283679,0,0,0.077987,0,0.317319,0,0.401917,0,0.356615,0.187727,0,0,0.000828,0,0.000064,0.702432,0,0,0.288186,0,0.396469,0,0.541582,0,0.332403,0.736288,0,0,0.001187,0,0.000276,0.205145,0,0.103314,0,0,0.343105,0,0.409840,0.593792,0,0,0.001891,0,0.001367,0,0.000085,0,0.028514,0,0.024192,0,0.130364,0,0.149696,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[1.004594031913367,0.872858870876535];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p64=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p64.svg"}], p64];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.764194,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.252295,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.048967,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.257761,0.216465,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.264116,0.076800,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.287958,0.251189,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.505696,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["65: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.634129041950018, 5.061752500851415];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p65=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p65.svg"}], p65];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.045868,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.229272,0,0.538458,0,0.427117,0,0,0.062437,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.551751,0,0,0.310208,0,0.459261,0,0.408545,0.755900,0,0,0.004560,0,0.000389,0,0.028086,0.083042,0,0,0.317319,0,0.401917,0,0.356615,0.031110,0,0.156280,0,0,0.000064,0,0.044627,0,0.005833,0,0.396469,0,0.541582,0,0.332403,0.034817,0,0.125528,0,0,0.000276,0,0.064708,0.594000,0,0,0.343105,0,0.473076,0,0.331698,0.795168,0,0,0.001367,0,0.000085,0,0.028514,0,0.117084,0.446596,0,0.227019,0,0.686754,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.035797351326896,-0.174878569813974];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p66=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p66.svg"}], p66];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "10.885704,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.090302,0,0.783556,0,0.643784,0,0.180700,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.762290,0,0.508420,0,0.374072,0,0.580671,0,0,0.011747,0,0.004560,0,0.000389,0,0.028086,0.680101,0,0,0.221976,0,0.401917,0.565934,0,0.750963,0,0,0.000828,0,0.000064,0,0.044627,0.700049,0,0,0.343282,0,0.541582,0.513185,0,0.808347,0,0,0.001187,0,0.000276,0,0.064708,0.736402,0,0.648071,0,0.422758,0,0.659479,0,0.574579,0,0,0.001367,0,0.000085,0,0.028514,0,0.146250,0.400028,0,0.596627,0,0.414941,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.192905108207474,-3.331986326694552];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p67=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p67.svg"}], p67];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "14.409365,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.055660,0.601627,0,0.340014,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.128641,0.340037,0,0,0.459261,0.450524,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.553140,0,0,0.145996,0,0.401917,0.435542,0,0,0.071096,0,0.000828,0,0.000064,0,0.044627,0.703971,0,0,0.349410,0,0.541582,0.659754,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.737383,0,0,0.144085,0.210993,0,0.383988,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.094290,0.541204,0,0,0.206314,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 2.036042249524603,-2.175123468011681];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p68=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p68.svg"}], p68];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.324663,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.587411,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.036467,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.395142,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.239790,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.402415,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.402510,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["69: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.569913374404222, 5.062242763955854];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p69=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p69.svg"}], p69];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.005249,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.392870,0,0,0.269942,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.217367,0,0.031286,0,0.133526,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.012546,0,0.267810,0,0.190397,0.105640,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.312304,0,0,0.396469,0,0.541582,0.066862,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.056010,0,0,0.343105,0,0.473076,0.346979,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.138407,0,0.177668,0,0.394549,0.441656,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 0;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["70: 0 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.003673028725123, 0.242458514655671];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p70=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p70.svg"}], p70];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.634569,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.064483,0.681529,0,0.215749,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.195553,0.537586,0,0,0.459261,0,0.408545,0.650753,0,0,0.004560,0,0.000389,0,0.028086,0.691866,0,0,0.270506,0,0.401917,0,0.215439,0.666404,0,0,0.000828,0,0.000064,0,0.044627,0.633628,0,0,0.396469,0,0.541582,0.574215,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.667039,0,0,0.343105,0.182562,0,0.388155,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.286593,0,0.549047,0,0.309127,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.906875747019062,-4.04595696550614 ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p71=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p71.svg"}], p71];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "11.000101,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.587411,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.485837,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.543181,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.164116,0.275084,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.160752,0.413199,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.178980,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["72: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.884619252900392, 5.915183800180219];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p72=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p72.svg"}], p72];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.010501,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.513391,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.338288,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.350044,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.312339,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.328150,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.296872,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["73: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.272364305476458, 5.042634802778028];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p73=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p73.svg"}], p73];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.080599,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.536921,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.511327,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.530436,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.170183,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.281340,0,0.047340,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.035726,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["74: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.94858980752042 , 4.344595628698133];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p74=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p74.svg"}], p74];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.711682,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.239617,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.349562,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.252614,0.221612,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.331763,0,0.071729,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.278399,0,0.133370,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.390659,0,0.079843,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["75: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.694178024451048, 3.247781979759   ];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p75=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p75.svg"}], p75];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.782836,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.315352,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.529955,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.586318,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.100880,0.373369,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.377905,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.320157,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["76: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.816236825804026, 6.525968092401288];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p76=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p76.svg"}], p76];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.148981,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.290352,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.365283,0,0.507651,0,0.583122,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.641259,0,0,0.401917,0.558091,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.200931,0,0.041812,0,0.541582,0.622499,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.064578,0,0.177419,0,0.042928,0,0.149590,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.126397,0.545126,0,0.097363,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.089407559032274,-0.408033053575994];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p77=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p77.svg"}], p77];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.726858,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.264862,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.503484,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.343426,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.026816,0,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.532385,0,0,0.278223,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.424783,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["78: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.01788628001127 , 1.878805061524192];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p78=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p78.svg"}], p78];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.061097,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.544340,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.362341,0,0.148827,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.092239,0,0.387054,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.430967,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.523003,0,0,0.249345,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.321137,0,0,0.045109,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["79: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.853915634494159, 2.693681402254541];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p79=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p79.svg"}], p79];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.934992,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.205945,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.459261,0.411798,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.334112,0,0,0.307351,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.047649,0,0,0.418053,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.181245,0.113268,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.675073,0,0,0.450953,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["80: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-3.455876509588096, 3.624073749530752];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p80=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p80.svg"}], p80];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.115376,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.269029,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.398582,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.062789,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.213641,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0,0.093664,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0,0.059255,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["81: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.733148673693449, 2.816899674852155];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p81=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p81.svg"}], p81];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.038474,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.380793,0,0,0.059496,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.292243,0,0,0.459261,0.085573,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.287546,0.207190,0,0,0.401917,0.040444,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.001912,0,0.381027,0,0.541582,0.538185,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.137383,0,0,0.343105,0,0.439987,0,0.014541,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.163897,0.094145,0,0.386088,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.059081586772213,-0.198162805259291];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p82=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p82.svg"}], p82];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.540735,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.830743,0,0.580370,0,0,0.318471,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.349545,0,0,0.462414,0.366964,0,0,0.190653,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.128723,0,0.317319,0,0.401917,0,0.123527,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.051421,0,0.396469,0,0.541582,0,0.332403,0.530896,0,0,0.001187,0,0.000276,0,0.064708,0.518755,0,0,0.343105,0,0.473076,0,0.133168,0.195168,0,0,0.001367,0,0.000085,0,0.028514,0,0.220515,0.385812,0,0.325549,0,0.620578,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[-0.003673028725123,-0.135408189761956];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p83=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p83.svg"}], p83];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "9.845747,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.060940,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.367700,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.339995,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.097464,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.168840,0,0.451017,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.255665,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["84: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.669178011040003, 1.683811313380264];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p84=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p84.svg"}], p84];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.283627,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.286676,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.301278,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.282025,0.419897,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.317057,0,0.198935,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.307811,0,0.297340,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.424482,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["85: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.589521171669275, 3.170821177680753];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p85=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p85.svg"}], p85];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.663718,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0.173752,0,0,0.034648,0,0.176408,0.432029,0,0,0.001175,0,0.000642,0,0.027970,0.759839,0,0,0.402119,0,0.459261,0,0.408545,0.788008,0,0,0.004560,0,0.000389,0.188826,0,0,0.133870,0,0.317319,0,0.401917,0,0.356615,0.182090,0,0,0.000828,0,0.000064,0.534785,0,0,0.263186,0,0.396469,0,0.541582,0,0.332403,0.171582,0,0,0.001187,0,0.000276,0,0.057355,0.706500,0,0,0.343105,0,0.473076,0,0.331698,0.890511,0,0,0.001367,0,0.000085,0,0.028514,0,0.088652,0.301988,0,0.326039,0,0.639940,0,0,0.008610,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.64388556009681 ,-0.782966778583888];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p86=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p86.svg"}], p86];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "11.837839,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.374911,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.380935,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.162789,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.275880,0,0.425406,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.222517,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.375953,0,0.379843,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["87: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.400239280354116, 1.930691419291933];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p87=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p87.svg"}], p87];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.822297,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.318471,0.374327,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.180340,0.027975,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.527005,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0.337110,0,0,0.541582,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.253464,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.328995,0,0,0.450953,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["88: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-4.023523635160062, 4.612309066701372];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p88=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p88.svg"}], p88];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "8.143806,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.232264,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.390739,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.406171,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0.279251,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.209282,0,0.114987,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.353949,0,0,0.394549,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["89: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-0.555942842880995, 2.873762431462072];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p89=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p89.svg"}], p89];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.528649,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.064483,0.431529,0,0,0.129349,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.023249,0.537586,0,0.540739,0,0.403220,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.691866,0,0,0.317319,0,0.401917,0.596326,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.414755,0,0,0.396469,0,0.541582,0.044068,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.260667,0,0,0.343105,0,0.473076,0.418302,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0.598357,0,0.549047,0,0.605451,0,0.731607,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.406630637818585,-1.545711856305663];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p90=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p90.svg"}], p90];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "10.253635,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.007999,0,0,0.083026,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.368296,0.080445,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.223202,0.051759,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.300145,0.234398,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.325458,0.426924,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.526039,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["91: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.399079984198124, 4.809301505704187];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p91=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p91.svg"}], p91];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.873491,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.671725,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.415004,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.448083,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.056763,0.154741,0,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.295130,0,0,0.292928,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.068900,0,0.069421,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["92: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.137560418019802, 3.162732894439958];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p92=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p92.svg"}], p92];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.223683,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.453833,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.267945,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.094407,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.046729,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.252905,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.394666,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["93: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.398344690124304, 3.722291710694097];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p93=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p93.svg"}], p93];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.451999,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.031478,0,0.368360,0,0.616087,0,0.053984,0,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.416702,0,0,0.425649,0,0.459261,0,0.130604,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0.610983,0,0,0.317319,0,0.401917,0,0.356615,0.431110,0,0,0.000828,0,0.000064,0,0.044627,0.370637,0,0,0.396469,0,0.541582,0,0.332403,0.696827,0,0,0.001187,0,0.000276,0,0.064708,0.597922,0,0,0.343105,0,0.473076,0.529577,0,0,0.022479,0,0.001367,0,0.000085,0,0.028514,0,0.197721,0.445861,0,0.568931,0,0.115676,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 0.761287555792937,-0.900368774280016];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p94=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p94.svg"}], p94];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "10.698936,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0,0.036118,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0,0.044555,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0,0.025446,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.396469,0,0.121729,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.343105,0.133542,0,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0,0.450953,0.407166,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["95: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-2.246629022205979, 3.583811408747696];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p95=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p95.svg"}], p95];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.710073,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.125156,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.462414,0.503484,0,0,0.408545,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0,0.317319,0.122103,0,0,0.356615,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.038380,0,0.395259,0,0.332403,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0.451013,0,0,0.473076,0,0.331698,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.264243,0,0,0.106314,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol};
predicted = Position[results,Max[results]][[1]][[1]] - 1;
totalCount += 1;
warning = "";
correctOutput = 1;
nnOutput = 1;
If[predicted == correctOutput, correctCount+=1, warning = "*"];
If[predicted == nnOutput, matchingCount+=1, warning = "**"];
Print["96: 1 " <> ToString[predicted] <> " 1 " <> warning];
correctResults=List[-1.903915720920894, 1.764834502433816];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p96=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p96.svg"}], p96];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.230015,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0.041772,0,0.124487,0,0.381529,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0.765231,0,0.529743,0,0.532896,0,0.583612,0,0.063744,0,0,0.004560,0,0.000389,0.239561,0,0.684022,0,0,0.226878,0,0.390888,0.370100,0,0.784786,0,0,0.000828,0,0.000064,0.223020,0,0.594412,0,0,0.396469,0,0.541582,0.180832,0,0.898053,0,0,0.001187,0,0.000276,0.202939,0,0.720961,0,0,0.219576,0.200944,0,0.660459,0,0.603256,0,0,0.001367,0,0.000085,0,0.028514,0.714289,0,0.541204,0,0.597608,0,0.207588,0,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 3.063738430613289,-3.202819649100367];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p97=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p97.svg"}], p97];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "13.608782,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.148307,0.215597,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0.511116,0,0,0.459261,0.482877,0,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.089262,0,0.102614,0,0.401917,0.279414,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0.488285,0,0,0.332743,0,0.541582,0.253871,0,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0.571206,0,0,0.251438,0,0.435576,0.340361,0,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.179339,0.463998,0,0.070157,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[ 1.264718892747174,-1.403800111234252];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p98=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p98.svg"}], p98];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "7.874085,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

rsys=MNISTReduced[0,-0.000000,0,0.000041,0,0.000490,0,0.003451,0,0.004257,0,0.001828,0,0.000187,0,-0.000000,0,0.000006,0,0.003693,0,0.051610,0,0.205660,0.203587,0,0,0.176408,0,0.034882,0,0.001175,0,0.000642,0,0.028705,0,0.226925,0,0.059227,0,0.459261,0,0.216388,0,0.099247,0,0.004560,0,0.000389,0,0.028086,0,0.308135,0.380965,0,0,0.401917,0.015934,0,0,0.083596,0,0.000828,0,0.000064,0,0.044627,0,0.288186,0,0.061175,0,0.541582,0,0.124805,0,0.094104,0,0.001187,0,0.000276,0,0.064708,0,0.254774,0,0.122517,0,0.439252,0,0.273119,0,0.063656,0,0.001367,0,0.000085,0,0.028514,0,0.230074,0.124537,0,0.353735,0,0,0.112265,0,0.010816,0,0.000202,0,-0.000000,0,0.000625,0,0.012744,0,0.034878,0,0.027462,0,0.006073,0,0.000506,0,0.000002];
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
correctResults=List[0.141848918242542,0.010113757205709];
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
p99=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1"}, PlotRange->All, AxesLabel->{"time","value"}];
Export[FileNameJoin[{"../../data-repo/kinetics/mnist-subset", "MNISTReduced_p99.svg"}], p99];

If[analyzeConvergenceTime, 
  outputConfidences = softmax[correctResults];
  correctOutputConfidence = outputConfidences[[correctOutput+1]];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[correctOutputConfidence,{20,15}]] <> ","];
  WriteString["MNISTReduced_convergence_times.csv", ToString[DecimalForm[Max[outputConfidences],{20,15}]] <> ","];
  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];
  For[i=1, i <= Length[times], i = i+1,
    WriteString["MNISTReduced_convergence_times.csv", DecimalForm[times[[i]], {20, 5}]];
    WriteString["MNISTReduced_convergence_times.csv", ","];
  ];
  WriteString["MNISTReduced_convergence_times.csv", "6.319996,"];
  plotterSummed={(hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol};
  summedOutputs = (plotterSummed /. {t->tmax})[[1]];
  WriteString["MNISTReduced_convergence_times.csv", DecimalForm[summedOutputs,{20,15}]];
  WriteLine["MNISTReduced_convergence_times.csv", ""];
];

Print["totalCount (number of predictions): " <> ToString[totalCount]];
Print["correctCount (number of correct predictions): " <> ToString[correctCount]];
Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];
