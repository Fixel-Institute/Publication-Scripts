#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:28:10 2022|

@author: jcagle
"""

import matplotlib.pyplot as plt
import numpy as np
import os, sys
import re
import copy
import pandas as pd
from matplotlib import ticker, colors
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from mplcursors import cursor
from scipy import stats, signal
import scipy.io as sio
from datetime import datetime

from statsmodels.stats import multitest

import statsmodels.formula.api as smf
import statsmodels.api as sm

import SignalProcessingUtility as SPU
import GraphingUtility as GU
import Percept
from PythonUtility import *

sys.path.append("D:\\fixel_academy\BRAVO\Server")
import BRAVOPlatformAPI

requester = BRAVOPlatformAPI.BRAVOPlatformRequest("Admin@ufl.edu","", server="http://10.16.15.166:3001")
PatientList = requester.RequestPatientList()

# %% Group Analysis
SenSightChronicLFP = []
UniquePatients = []
for patient in PatientList:
    print(patient["ID"])
    PatientInfo = requester.RequestPatientInfo(patient["ID"])
    if not PatientInfo["FirstName"] == "":
        ChronicLFP = requester.RequestChronicLFP(patient["ID"])
        Events = requester.RequestPatientEvents(patient["ID"])
        for device in PatientInfo["Devices"]:
            for lead in device["Leads"]:
                for data in ChronicLFP["ChronicData"]:
                    if "Hemisphere" in data.keys() and data["Device"] == device["DeviceName"]:
                        if data["Hemisphere"] == lead["TargetLocation"]:
                            UniquePatients.append(patient["ID"])
                            PatientInfo["ID"] = patient["ID"]
                            data["PatientInfo"] = PatientInfo
                            for i in range(len(Events["EventPSDs"])):
                                if Events["EventPSDs"][i]["Device"] == device["DeviceName"]:
                                    data["Events"] = {
                                        "EventName": Events["EventPSDs"][i]["EventName"],
                                        "EventTime": Events["EventPSDs"][i]["EventTime"]
                                    }
                            SenSightChronicLFP.append(data)
    
# %% Inform Deidentification
SubtypeTable = pd.read_excel("PD_INFORM_IdentificationTable.xlsx")
CircadianTable = []

MissingInformID = []
Count = 0
for i in range(len(SenSightChronicLFP)):
    FoundMRN = ""
    SenSightChronicLFP[i]["PatientInfo"]["MRN"] = re.sub('[^0-9]','', SenSightChronicLFP[i]["PatientInfo"]["MRN"])
    if not SenSightChronicLFP[i]["PatientInfo"]["MRN"] == "":
        for j in range(len(SubtypeTable["MRN"])):
            if SubtypeTable["MRN"][j] == int(SenSightChronicLFP[i]["PatientInfo"]["MRN"]):
                FoundMRN = SubtypeTable["idPatient"][j]
                break
    
    if FoundMRN == "":
        Count += 1
        MissingInformID.append(SenSightChronicLFP[i]["PatientInfo"]["MRN"])
        SenSightChronicLFP[i]["INFORM"] = ""
        CircadianTable.append(SenSightChronicLFP[i])
        
    else:
        SenSightChronicLFP[i]["INFORM"] = FoundMRN
        CircadianTable.append(SenSightChronicLFP[i])

# %% Calculate Circadian Parameters
CircadianResult = []
for ChronicLFP in CircadianTable:
    if ChronicLFP["Hemisphere"].startswith("Right"):
        TherapyHemisphere = "RightHemisphere"
    else:
        TherapyHemisphere = "LeftHemisphere"

    ImplantDates = [device["ImplantDate"] for device in ChronicLFP["PatientInfo"]["Devices"]]
    ImplantDates.sort()
    if np.any(np.diff(ImplantDates) < 3600*24):
        print(f"{ChronicLFP['PatientInfo']['FirstName']} {ChronicLFP['PatientInfo']['LastName']} Multi-Device Patient")
        continue
    
    ImplantTime = 0
    ImplantDevice = None
    for device in ChronicLFP["PatientInfo"]["Devices"]:
        if device["DeviceName"] == ChronicLFP["Device"]:
            ImplantTime = device["ImplantDate"]
            ImplantDevice = device
    
    LeadType = ""
    for lead in ImplantDevice["Leads"]:
        if lead["TargetLocation"] == ChronicLFP["Hemisphere"]:
            LeadType = lead["ElectrodeType"]
    
    ReplacementBattery = False
    for device in ChronicLFP["PatientInfo"]["Devices"]:
        if device["ImplantDate"] < ImplantTime:
            ReplacementBattery = True
    
    Hemisphere = []
    UniqueTherapyOption = []
    for segment in range(len(ChronicLFP["Timestamp"])):
        if not ChronicLFP["Therapy"][segment]["TherapyOverview"] in UniqueTherapyOption:
            UniqueTherapyOption.append(ChronicLFP["Therapy"][segment]["TherapyOverview"])
    
    for therapyOption in UniqueTherapyOption:
        if TherapyHemisphere in Hemisphere:
            continue
        
        ChronicPower = []
        ChronicAmplitude = []
        ChronicEvents = []
        ChronicTimestamps = []
        for segment in range(len(ChronicLFP["Timestamp"])):
            if ChronicLFP["Therapy"][segment]["TherapyOverview"] == therapyOption:
                SegmentEvents =  [datetime.fromtimestamp(ChronicLFP["EventTime"][segment][i]) for i in range(len(ChronicLFP["EventTime"][segment]))] 
                Timestamps = copy.deepcopy(ChronicLFP["Timestamp"][segment])
                ChronicEvents.extend(SegmentEvents)
                ChronicPower.extend(ChronicLFP["Power"][segment])
                ChronicAmplitude.extend(ChronicLFP["Amplitude"][segment])
                ChronicTimestamps.extend(Timestamps)
                ChronicTherapy = ChronicLFP["Therapy"][segment]
        ChronicPower = np.array(ChronicPower)
        ChronicTimestamps = np.array(ChronicTimestamps)
        
        try:
            SensingFrequency = ChronicTherapy[TherapyHemisphere]["SensingSetup"]["FrequencyInHertz"]
        except:
            SensingFrequency = -1
            
        if SensingFrequency > 30 or SensingFrequency < 7:
            continue
            pass
        
        if len(ChronicPower) == 0:
            continue
        
        TotalKeep = 24*6*5
        
        for t in range(len(ChronicAmplitude)):
            currentWindow = ChronicAmplitude[t:t+TotalKeep]
            TimeUpdate = np.diff(ChronicTimestamps[t:t+TotalKeep])
            if len(TimeUpdate) == 0:
                continue 
            
            if np.max(currentWindow)-np.min(currentWindow) < 0.5 and np.max(TimeUpdate) < 3600:
                ChronicPower = ChronicPower[t:t+TotalKeep]
                ChronicTimestamps = [ChronicTimestamps[i] for i in range(len(ChronicTimestamps)) if i < t+TotalKeep and i >= t]
                ChronicEvents = [ChronicEvents[i] for i in range(len(ChronicEvents)) if ChronicEvents[i].timestamp() > ChronicTimestamps[0] and ChronicEvents[i].timestamp() < ChronicTimestamps[-1]]
                break
        
        TargetChannel = [-1,-1]
        Impedance = 0
        TherapyAmplitude = -1
        try:
            if "E2" in ChronicTherapy[TherapyHemisphere]["Channel"] and "E1" in ChronicTherapy[TherapyHemisphere]["Channel"]:
                TargetChannel = [0,3]
                if TherapyHemisphere.replace("Hemisphere","") in ChronicTherapy["Impedance"]["log"].keys():    
                    Impedance = ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][0][-1]
            
            elif "E2" in ChronicTherapy[TherapyHemisphere]["Channel"]:
                TargetChannel = [1,3]
                if TherapyHemisphere.replace("Hemisphere","") in ChronicTherapy["Impedance"]["log"].keys():
                    if len(ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"]) == 4:
                        Impedance = ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][1][3]
                    else:
                        Impedance = ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][1][7]
                        Impedance += ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][2][7]
                        Impedance += ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][3][7]
                        Impedance /= 3
                        
            elif "E1" in ChronicTherapy[TherapyHemisphere]["Channel"]:
                TargetChannel = [0,2]
                if TherapyHemisphere.replace("Hemisphere","") in ChronicTherapy["Impedance"]["log"].keys():
                    if len(ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"]) == 4:
                        Impedance = ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][0][2]
                    else:
                        Impedance = ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][0][4]
                        Impedance += ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][0][5]
                        Impedance += ChronicTherapy["Impedance"]["log"][TherapyHemisphere.replace("Hemisphere","")]["Bipolar"][0][6]
                        Impedance /= 3
            TherapyAmplitude = ChronicTherapy[TherapyHemisphere]["Amplitude"]
        except:
            pass
            
        ChronicTimestamps = [datetime.fromtimestamp(ChronicTimestamps[i]) for i in range(len(ChronicTimestamps))]
        RealTime = np.array([t.timestamp() for t in ChronicTimestamps])
        CircadianTimestamp = np.array([t.hour + np.round((t.minute/60)*6)/6 for t in ChronicTimestamps])
        EventTimestamp = np.array([t.hour + np.round((t.minute/60)*6)/6 for t in ChronicEvents])
            
        Power = np.array(ChronicPower)
        Power[Power > 1e6] = 1e6
            
        RealTime = RealTime[Power > 0]
        CircadianTimestamp = CircadianTimestamp[Power > 0]
        Power = Power[Power > 0]
            
        if len(Power) != TotalKeep:
            continue
        
        normPower = np.zeros(Power.shape)
        for i in range(len(Power)):
            refPower = 0
            PeriodSelection = rangeSelection(RealTime, [RealTime[i]-12*3600, RealTime[i]+12*3600])
            while np.sum(PeriodSelection) < 3:
                refPower += 1
                PeriodSelection = rangeSelection(RealTime, [RealTime[i]-12*refPower*3600, RealTime[i]+12*refPower*3600])
                
            normPower[i] = (Power[i] - np.mean(Power[PeriodSelection])) / np.std(Power[PeriodSelection])
            
            if not np.any(PeriodSelection):
                raise Exception("Not Enough Data")
        
        RawPower = copy.deepcopy(Power)
        Power = normPower
            
        HourClock = np.arange(0,24*6)/6
        MeanFunction = np.zeros(HourClock.shape)
        EventHistogram = np.zeros(HourClock.shape)
        
        for i in range(len(HourClock)):
            SelectedTimestamps = rangeSelection(CircadianTimestamp, [HourClock[i]-1/12, HourClock[i]+3/12], "inclusive")
            EventHistogram[i] = np.sum(rangeSelection(EventTimestamp, [HourClock[i]-1/12, HourClock[i]+3/12], "inclusive"))
            if np.any(SelectedTimestamps):
                MeanFunction[i] = np.mean(Power[SelectedTimestamps])
                    
        NightPeriod = [0,5]
        DayPeriod = [15,20]
        NightPower = Power[rangeSelection(CircadianTimestamp, NightPeriod, "inclusive")]
        DayPower = Power[rangeSelection(CircadianTimestamp, DayPeriod, "inclusive")]
        
        EventTimestamp.sort()
        if len(EventTimestamp) > 2:
            EventPeriod = [EventTimestamp[0], EventTimestamp[-1]]
        else:
            EventPeriod = [0,0]
        EventsInDay = rangeSelection(EventTimestamp, DayPeriod)
        EventsInNight = rangeSelection(EventTimestamp, NightPeriod)
        
        Hemisphere.append(TherapyHemisphere)
        CircadianResult.append({
            "PatientID": ChronicLFP["PatientInfo"]["ID"],
            "INFORM": ChronicLFP["INFORM"],
            "MRN": ChronicLFP["PatientInfo"]["MRN"],
            "DeviceID": ImplantDevice["ID"],
            "Diagnosis": ChronicLFP["PatientInfo"]["Diagnosis"],
            "LeadType": LeadType.split(" ")[0],
            "Hemisphere": TherapyHemisphere,
            "Target": ChronicLFP["Hemisphere"].split(" ")[1],
            "TherapyOption": therapyOption,
            "TherapyFrequency": ChronicTherapy[TherapyHemisphere]["Frequency"],
            "TherapyPulseWidth": ChronicTherapy[TherapyHemisphere]["PulseWidth"],
            "TherapyAmplitude": TherapyAmplitude,
            "TherapyContact": str(TargetChannel),
            "TherapyImpedance": Impedance,
            "SensingFrequency": SensingFrequency,
            "ImplantTime": ImplantTime,
            "RecordingTime": ChronicTimestamps[0],
            "SampleSize": len(Power)/144,
            "StartTime": (RealTime[0] - ImplantTime) / 3600 / 24,
            "DayPower": np.mean(DayPower),
            "NightPower": np.mean(NightPower),
            "CircadianTrend": MeanFunction,
            "EventHistogram": EventHistogram,
            "Statistics": stats.ttest_ind(DayPower, NightPower).pvalue,
        })
        
# %% Unique Patient Information
CircadianResult = pd.DataFrame(CircadianResult)
UniquePatients = np.unique(CircadianResult["PatientID"])

Count = 0
MissingInfo = []
INFORMs = []
UniquePatientIDs = []
UniquePatientResults = []
for patient in UniquePatients:
    PatientResult = CircadianResult.loc[CircadianResult["PatientID"] == patient]
    if len(np.unique(PatientResult["DeviceID"])) > 1:
        print(patient + " Multiple Devices")
        #continue
    
    Label = -1
    for informRow in SubtypeTable.index:
        if SubtypeTable["idPatient"][informRow] == np.unique(PatientResult["INFORM"])[0]:
            Label = informRow
    
    if not type(Label) == int:
        Label = -1
        
    if Label == -1:
        INFORMs.append(np.unique(PatientResult["INFORM"])[0])
     
    # Skip Results without Sensing Frequency
    PatientResult = PatientResult.loc[PatientResult["SensingFrequency"] > 0]
    
    UniqueHemisphere = []
    for hemisphere in np.unique(PatientResult["Hemisphere"]):
        for channel in np.unique(PatientResult["TherapyContact"]):
            for senseFreq in np.unique(PatientResult["SensingFrequency"]):
                if senseFreq > 0 and senseFreq < 80:
                    SelectedResult = (PatientResult["SensingFrequency"] == senseFreq) & (PatientResult["TherapyContact"] == channel) & (PatientResult["Hemisphere"] == hemisphere)
                    if np.any(SelectedResult):
                        SubResults = PatientResult.loc[SelectedResult]
                        
                        if len(SubResults) > 1:
                            MissingInfo.append(patient)
                            Count += 1
                        
                        DayPowerChange = np.median(SubResults["DayPower"] - SubResults["NightPower"])
                        RawDayPower = np.mean(SubResults["DayPower"])
                        RawNightPower = np.mean(SubResults["NightPower"])
                        
                        if not patient in UniquePatientIDs:
                            UniquePatientIDs.append(patient)
                        
                        UniquePatientResults.append({
                            "PatientID": patient,
                            "INFORM": np.unique(SubResults["INFORM"])[0],
                            "MRN": np.unique(SubResults["MRN"])[0],
                            "Diagnosis": np.unique(SubResults["Diagnosis"])[0],
                            "Hemisphere": hemisphere,
                            "Target": np.unique(SubResults["Target"])[0],
                            "LeadType": np.unique(SubResults["LeadType"])[0],
                            "ImplantTime": datetime.fromtimestamp(np.unique(SubResults["ImplantTime"])[0]),
                            "RecordingTime": np.unique(SubResults.loc[PatientResult["Hemisphere"] == hemisphere]["RecordingTime"])[0],
                            "RecordingDuration": np.unique(SubResults["SampleSize"])[0],
                            "Subtype": Label,
                            "LabelIndex": Label,
                            "Channel": channel,
                            "SensingFrequency": senseFreq,
                            "TherapyOption": np.unique(SubResults["TherapyOption"])[0],
                            "TherapyImpedance": np.unique(SubResults["TherapyImpedance"])[0],
                            "TherapyAmplitude": np.unique(SubResults["TherapyAmplitude"])[0],
                            "TherapyPulseWidth": np.unique(SubResults["TherapyPulseWidth"])[0],
                            "TherapyFrequency": np.unique(SubResults["TherapyFrequency"])[0],
                            "DayPowerChange": DayPowerChange,
                            "RawDayPower": RawDayPower,
                            "RawNightPower": RawNightPower,
                            "Statistics": np.mean(SubResults["Statistics"]),
                            "EventTrend": np.sum(SubResults["EventHistogram"].sum()),
                            "EventHistogram": np.sum(SubResults["EventHistogram"]),
                            "CircadianTrend": np.mean(SubResults["CircadianTrend"]),
                        })

UniquePatientResults = pd.DataFrame(UniquePatientResults)

# %% Graphing Power vs Sensing Frequency
GraphingResult = UniquePatientResults.loc[(UniquePatientResults["Diagnosis"] == "ParkinsonsDisease") & 
                                          (UniquePatientResults["MRN"] != "") & 
                                          ((UniquePatientResults["Target"] == "STN")) & 
                                          (UniquePatientResults["SensingFrequency"] >= 0) & 
                                          (UniquePatientResults["SensingFrequency"] < 90) & 
                                          (UniquePatientResults["Subtype"] != -2) & 
                                          (UniquePatientResults["LeadType"] != "")]

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

r, p = stats.pearsonr(GraphingResult["SensingFrequency"], GraphingResult["DayPowerChange"])

ax.scatter(GraphingResult[GraphingResult["Target"] == "GPi"]["SensingFrequency"], 
           GraphingResult[GraphingResult["Target"] == "GPi"]["DayPowerChange"], 20, color="k")
ax.scatter(GraphingResult[GraphingResult["Target"] == "STN"]["SensingFrequency"], 
           GraphingResult[GraphingResult["Target"] == "STN"]["DayPowerChange"], 20, color="r")
ax.scatter(GraphingResult[GraphingResult["Target"] == "VIM"]["SensingFrequency"], 
           GraphingResult[GraphingResult["Target"] == "VIM"]["DayPowerChange"], 20, color="b")
ax.plot([0,100],[0,0], "k", linewidth=2)
ax.set_xlabel("Sensing Frequency", fontsize=15)
ax.set_ylabel("Daytime Power Changes (zscore)", fontsize=15)
ax.set_title("Sensing Frequency vs Power Changes", fontsize=21)
ax.legend(["GPi", "STN", "VIM"], fontsize=18, frameon=True, loc="lower right")
ax.set_ylim(-5,5)
ax.set_xlim(0, 40)

fig.savefig("SensingFrequencyScatterPlot80.eps")

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

GU.singleViolin(0, GraphingResult.loc[GraphingResult["SensingFrequency"] < 13]["DayPowerChange"], width=0.3, showmeans=True, ax=ax)
GU.singleViolin(1, GraphingResult.loc[(GraphingResult["SensingFrequency"] < 20) & 
                                      (GraphingResult["SensingFrequency"] >= 13)]["DayPowerChange"], width=0.3, showmeans=True, ax=ax)
GU.singleViolin(2, GraphingResult.loc[(GraphingResult["SensingFrequency"] < 30) & 
                                      (GraphingResult["SensingFrequency"] >= 20)]["DayPowerChange"], width=0.3, showmeans=True, ax=ax)
#GU.singleViolin(3, GraphingResult.loc[(GraphingResult["SensingFrequency"] <= 90) & 
#                                      (GraphingResult["SensingFrequency"] >= 30)]["DayPowerChange"], width=0.3, showmeans=True, ax=ax)

ax.plot([-1,4],[0,0], "k", linewidth=2)

ax.set_xticks([0,1,2])
ax.set_xticklabels([f"Theta/Alpha", 
                    f"Low Beta", 
                    f"High Beta"], fontdict={"fontsize": 15})

ax.set_ylabel("Daytime Power Changes (zscore)", fontsize=15)
ax.set_xlim(-1,3)
ax.set_ylim(-5,5)
fig.savefig("SensingFrequencyBarPlot.eps")

Alpha = GraphingResult.loc[GraphingResult["SensingFrequency"] < 13]["DayPowerChange"]
LBeta = GraphingResult.loc[(GraphingResult["SensingFrequency"] < 20) & 
                           (GraphingResult["SensingFrequency"] >= 13)]["DayPowerChange"]
HBeta = GraphingResult.loc[(GraphingResult["SensingFrequency"] < 30) & 
                           (GraphingResult["SensingFrequency"] >= 20)]["DayPowerChange"]
Gamma = GraphingResult.loc[(GraphingResult["SensingFrequency"] <= 90) & 
                           (GraphingResult["SensingFrequency"] >= 30)]["DayPowerChange"]

print(stats.f_oneway(Alpha, LBeta, HBeta))
print(stats.tukey_hsd(Alpha, LBeta, HBeta))   

AlphaP = stats.ttest_1samp(Alpha, 0).pvalue
LBetaP = stats.ttest_1samp(LBeta, 0).pvalue
HBetaP = stats.ttest_1samp(HBeta, 0).pvalue
GammaP = stats.ttest_1samp(Gamma, 0).pvalue
print(multitest.fdrcorrection([AlphaP, LBetaP, HBetaP], 0.05)[1])
print(np.array([AlphaP, LBetaP, HBetaP])*3)

# %% CircadianTrendColorHeatmap
GraphingResult = UniquePatientResults.loc[(UniquePatientResults["Diagnosis"] == "ParkinsonsDisease") & 
                                          (UniquePatientResults["MRN"] != "") & 
                                          (UniquePatientResults["Target"] == "GPi") & 
                                          (UniquePatientResults["SensingFrequency"] >= 0) & 
                                          (UniquePatientResults["SensingFrequency"] < 90) & 
                                          (UniquePatientResults["Subtype"] != -2) & 
                                          (UniquePatientResults["LeadType"] != "")]

GraphingResult["PowerBand"] = 0
GraphingResult.loc[(GraphingResult["SensingFrequency"] >= 13) & 
                   (GraphingResult["SensingFrequency"] < 21),"PowerBand"] = 1
GraphingResult.loc[(GraphingResult["SensingFrequency"] >= 21) & 
                   (GraphingResult["SensingFrequency"] < 30),"PowerBand"] = 2
GraphingResult.loc[(GraphingResult["SensingFrequency"] >= 30) & 
                   (GraphingResult["SensingFrequency"] < 90),"PowerBand"] = 3

GraphingResult = GraphingResult.sort_values(by=["PowerBand", "DayPowerChange"]).reset_index()

HourClock = np.arange(0,24*6)/6
Heatmap = np.zeros((len(HourClock), len(GraphingResult)))
for i in GraphingResult.index:
    Heatmap[:,i] = GraphingResult["CircadianTrend"][i]
#Heatmap = np.log10(Heatmap)

fig = GU.largeFigure(0, [800,1000])
ax = GU.addAxes(fig)

image = GU.imagesc(HourClock, range(len(GraphingResult)), Heatmap.T, clim=[-1,1], interpolation="antialiased")
GU.addColorbar(ax, image, "Normalized Power")

ax.plot([-100, 100], np.sum(GraphingResult["PowerBand"] <= 0) * np.ones(2), "k", linewidth=2)
ax.plot([-100, 100], np.sum(GraphingResult["PowerBand"] <= 1) * np.ones(2), "k", linewidth=2)
ax.plot([-100, 100], np.sum(GraphingResult["PowerBand"] <= 2) * np.ones(2), "k", linewidth=2)
ax.set_xlim(0,23.5)
ax.set_ylim(0, len(GraphingResult)-1)

ax.set_xlabel("Circadian Clock (hr)", fontsize=15)
ax.set_ylabel("Unique Recordings", fontsize=15)

fig.savefig("CircadianRhythmHeatmap.eps")

fig = GU.largeFigure(0, [800,200])
ax = GU.addAxes(fig)

ax.bar(HourClock, np.sum(GraphingResult["EventHistogram"]), width=0.16, edgecolor="k")
ax.set_xlim(0,24)
ax.set_ylim(0,40)

ax.set_xlabel("Circadian Clock (hr)", fontsize=15)
ax.set_ylabel("Events", fontsize=15)

fig.savefig("CircadianRhythmEventbars.eps")

# %% GLM For Medication
pd.options.mode.chained_assignment = None

LEDDTable = pd.read_excel("Medication Database.xlsx")
UPDRSTable = pd.read_excel("SubtypeMeasurements_UPDRS_left_right.xlsx", "UPDRS_Transpose")
MedicationTable = pd.read_excel("MedicationER.xlsx")


GraphingResult = UniquePatientResults.loc[(UniquePatientResults["Diagnosis"] == "ParkinsonsDisease") & 
                                          (UniquePatientResults["MRN"] != "") & 
                                          ((UniquePatientResults["Target"] == "GPi")) & 
                                          (UniquePatientResults["SensingFrequency"] >= 13) & 
                                          (UniquePatientResults["SensingFrequency"] < 30) & 
                                          (UniquePatientResults["Subtype"] != -2) & 
                                          (UniquePatientResults["LeadType"] != "")]

GraphingResult["SleepMedication"] = 0
GraphingResult["LEDD"] = 0
GraphingResult["Levodopa"] = 0
GraphingResult["Levodopa ER"] = 0
GraphingResult["Levodopa ER Nighttime"] = 0
GraphingResult["PowerBand"] = 0
GraphingResult["Laterality"] = 0
GraphingResult["UPDRS"] = 0

GraphingResult.loc[(GraphingResult["SensingFrequency"] >= 13) & 
                   (GraphingResult["SensingFrequency"] < 21),"PowerBand"] = 1
GraphingResult.loc[(GraphingResult["SensingFrequency"] >= 21) & 
                   (GraphingResult["SensingFrequency"] < 30),"PowerBand"] = 2
GraphingResult.loc[(GraphingResult["SensingFrequency"] >= 30) & 
                   (GraphingResult["SensingFrequency"] < 90),"PowerBand"] = 3

for i in GraphingResult.index:
    FoundLabel = -1
    for j in LEDDTable.index:
        if not GraphingResult["INFORM"][i] == "":
            if int(LEDDTable["Inform ID"][j]) == int(GraphingResult["INFORM"][i]):
                FoundLabel = j
                break
    
    if FoundLabel >= 0:
        if LEDDTable["Benzodiazepines"][FoundLabel] == "Yes" or LEDDTable["Melatoninergics"][FoundLabel] == "Yes":
            GraphingResult["SleepMedication"][i] = 1
        
        GraphingResult["LEDD"][i] = LEDDTable["LEED"][FoundLabel]
        
        PatientInfo = requester.RequestPatientInfo(GraphingResult["PatientID"][i])
        GraphingResult["Laterality"][i] = 0
        for k in range(len(PatientInfo["Devices"])):
            if len(PatientInfo["Devices"][k]["Leads"]) > GraphingResult["Laterality"][i]:
                GraphingResult["Laterality"][i] = len(PatientInfo["Devices"][k]["Leads"])
           
    else:
        print(GraphingResult["MRN"][i])
        Count += 1
        
GraphingResult["SubtypeRatio"] = 0
for i in GraphingResult.index:
    FoundLabel = -1
    for j in UPDRSTable.index:
        if not GraphingResult["INFORM"][i] == "":
            if int(UPDRSTable["MRN"][j]) == int(GraphingResult["MRN"][i]):
                FoundLabel = j
                break
    
    if FoundLabel >= 0:
        if GraphingResult["Hemisphere"][i] == "LeftHemisphere":
            GraphingResult["SubtypeRatio"][i] = UPDRSTable["Average Tremor Right"][FoundLabel] / UPDRSTable["Average AR Right"][FoundLabel]
        else:
            GraphingResult["SubtypeRatio"][i] = UPDRSTable["Average Tremor Left"][FoundLabel] / UPDRSTable["Average AR Left"][FoundLabel]
        
        GraphingResult["SubtypeRatio"][i] = UPDRSTable["Kang's Result"][FoundLabel]
        GraphingResult["UPDRS"][i] = UPDRSTable["Total motor score (UPDRS-III)"][FoundLabel]
        
    else:
        print(GraphingResult["PatientID"][i])
        
for i in GraphingResult.index:
    FoundLabel = -1
    for j in MedicationTable.index:
        if not GraphingResult["INFORM"][i] == "":
            if MedicationTable["PatientID"][j] == GraphingResult["PatientID"][i]:
                FoundLabel = j
                break
    
    if FoundLabel >= 0:
        GraphingResult["Levodopa"][i] = int(MedicationTable["DaytimeTablet"][FoundLabel])
        GraphingResult["Levodopa ER Nighttime"][i] = int(MedicationTable["ExtendedNighttime"][FoundLabel])
        if not MedicationTable["ExtendedNighttime"][FoundLabel] and MedicationTable["MultipleTabletsPerDay(ER)"][FoundLabel] > 2:
            GraphingResult["Levodopa ER Nighttime"][i] = 1
        if not MedicationTable["ExtendedNighttime"][FoundLabel] and MedicationTable["MultipleTabletsPerDay(ER)"][FoundLabel] > 2:
            GraphingResult["Levodopa ER Nighttime"][i] = 1
    else:
        print(GraphingResult["PatientID"][i])
        
GraphingResult["TEED"] = GraphingResult["TherapyPulseWidth"]/1e6 * \
               GraphingResult["TherapyFrequency"] * \
               GraphingResult["TherapyAmplitude"] / 1e3 * \
                GraphingResult["TherapyImpedance"] * \
                GraphingResult["TherapyAmplitude"] / 1e3
                
exog = GraphingResult[["Channel", "TEED", "LEDD", "Levodopa ER Nighttime", "SleepMedication", "SubtypeRatio", "UPDRS"]]

exog["Channel"] = np.array(exog["Channel"] == "[1, 3]", dtype=int)
exog = sm.add_constant(exog)
endog = GraphingResult["DayPowerChange"]

linearModel = sm.GLM(endog, exog)
linearModelResult = linearModel.fit()
print(linearModelResult.summary().as_csv())

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

GU.singleViolin(0, endog[exog["Levodopa ER Nighttime"] == 0], color="b", width=0.3, showmeans=True, ax=ax)
GU.singleViolin(1, endog[exog["Levodopa ER Nighttime"] == 1], color="r", width=0.3, showmeans=True, ax=ax)

p = stats.ttest_ind(endog[exog["Levodopa ER Nighttime"] == 0], endog[exog["Levodopa ER Nighttime"] == 1]).pvalue

ax.set_xticks([0,1])
ax.set_xticklabels([f"No Levodopa ER (n={np.sum(exog['Levodopa ER Nighttime'] == 0)})", 
                    f"Levodopa ER (n={np.sum(exog['Levodopa ER Nighttime'] == 1)})"], fontdict={"fontsize": 15}, rotation=10)
ax.set_xlim(-1,2)
ax.set_ylim(-3,3)
ax.set_title(f"Levodopa ER Comparison ({p:.3f})", fontsize=21)

fig.savefig("CircadianRhythmExtendedReleaseBetaBand.eps")

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

GU.singleViolin(0, endog[exog["SleepMedication"] == 0], color="b", width=0.3, showmeans=True, ax=ax)
GU.singleViolin(1, endog[exog["SleepMedication"] == 1], color="r", width=0.3, showmeans=True, ax=ax)

p = stats.ttest_ind(endog[exog["SleepMedication"] == 0], endog[exog["SleepMedication"] == 1]).pvalue

ax.set_xticks([0,1])
ax.set_xticklabels([f"No Benzo (n={np.sum(exog['SleepMedication'] == 0)})", 
                    f"Benzo (n={np.sum(exog['SleepMedication'] == 1)})"], fontdict={"fontsize": 15}, rotation=10)
ax.set_xlim(-1,2)
ax.set_ylim(-3,3)
ax.set_title(f"SleepMedication Comparison ({p:.3f})", fontsize=21)

fig.savefig("CircadianRhythmBenzoBetaBand.eps")

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

ax.scatter(exog["SubtypeRatio"], endog, 5, color="k")
p = stats.pearsonr(exog["SubtypeRatio"], endog).pvalue

ax.set_xlabel("Subtype Ratio (TD/AR)", fontsize=15)
ax.set_ylabel("Circadian Rhythm (z-score)", fontsize=15)
ax.set_ylim(-3,3)
ax.set_title(f"Subtype Comparison ({p:.3f})", fontsize=21)

fig.savefig("CircadianRhythmSubtype.eps")

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

ax.scatter(GraphingResult["UPDRS"], GraphingResult["DayPowerChange"], 5, color="k")
p = stats.pearsonr(GraphingResult["UPDRS"], GraphingResult["DayPowerChange"]).pvalue

ax.set_xlabel("UPDRS", fontsize=15)
ax.set_ylabel("Circadian Rhythm (z-score)", fontsize=15)
ax.set_ylim(-3,3)
ax.set_title(f"UPDRS Comparison ({p:.3f})", fontsize=21)

fig.savefig("CircadianRhythmUPDRS.eps")

fig = GU.largeFigure(0, [800,600])
ax = GU.addAxes(fig)

ax.scatter(GraphingResult[(GraphingResult["Target"] == "GPi") &
                          (GraphingResult["Levodopa ER Nighttime"] == 1)]["SensingFrequency"], 
           GraphingResult[(GraphingResult["Target"] == "GPi") &
                          (GraphingResult["Levodopa ER Nighttime"] == 1)]["DayPowerChange"], 30, color="r")
ax.scatter(GraphingResult[(GraphingResult["Target"] == "GPi") &
                          (GraphingResult["Levodopa ER Nighttime"] == 0)]["SensingFrequency"], 
           GraphingResult[(GraphingResult["Target"] == "GPi") &
                          (GraphingResult["Levodopa ER Nighttime"] == 0)]["DayPowerChange"], 30, color="b")
ax.plot([0,100],[0,0], "k", linewidth=2)
ax.set_xlabel("Sensing Frequency", fontsize=15)
ax.set_ylabel("Daytime Power Changes (zscore)", fontsize=15)
ax.set_title("Patients with Nighttime Extended-Released L-Dopa", fontsize=21)
#ax.legend(["GPi"], fontsize=18, frameon=True, loc="lower right")
ax.set_ylim(-5,5)
ax.set_xlim(10, 30)
fig.savefig("CircadianRhythmExtendedReleaseScatter.eps")
