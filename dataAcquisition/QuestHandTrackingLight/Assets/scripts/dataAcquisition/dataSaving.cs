using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Net;
using System.Net.Sockets;
using System;
using System.Diagnostics;

public class dataSaving : MonoBehaviour
{
    /*
     Waits for remote command to start recording motion capture of the hands
     - When a "Start Recording" signal arrives, writes the motion capture data in the given file
     - Stops when the "Stop Recording" signal arrives
     Displays to the user a witness of the recording state
         */


    public Transform handContainer;
    public OVRSkeleton leftHand;
    public OVRSkeleton rightHand;
    public GameObject savingDisplay;
    public Material savingMaterialOn;
    public Material savingMaterialOff;
    public GestureRecognizer gestureRecognizerRight;
    public GestureRecognizer gestureRecognizerLeft;
    public gestureDisplay gestureDisplay;

    string path;
    bool isSaving = false;
    string currentRecordingName = "";
    bool sendFeedBack = false;
    
    public TextMesh textDisplay;
    public Transform elbow;
    public Transform arm;
    


    // source :  https://stackoverflow.com/questions/1193955/how-to-query-an-ntp-server-using-c
    public static DateTime GetNetworkTime()
    {
    const string NtpServer = "be.pool.ntp.org";

    const int DaysTo1900 = 1900 * 365 + 95; // 95 = offset for leap-years etc.
    const long TicksPerSecond = 10000000L;
    const long TicksPerDay = 24 * 60 * 60 * TicksPerSecond;
    const long TicksTo1900 = DaysTo1900 * TicksPerDay;

    var ntpData = new byte[48];
    ntpData[0] = 0x1B; // LeapIndicator = 0 (no warning), VersionNum = 3 (IPv4 only), Mode = 3 (Client Mode)

    var addresses = Dns.GetHostEntry(NtpServer).AddressList;
    var ipEndPoint = new IPEndPoint(addresses[0], 123);
    long pingDuration = Stopwatch.GetTimestamp(); // temp access (JIT-Compiler need some time at first call)
    using (var socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp))
    {
        socket.Connect(ipEndPoint);
        socket.ReceiveTimeout = 3000;
        socket.Send(ntpData);
        pingDuration = Stopwatch.GetTimestamp(); // after Send-Method to reduce WinSocket API-Call time

        socket.Receive(ntpData);
        pingDuration = Stopwatch.GetTimestamp() - pingDuration;
    }

    long pingTicks = pingDuration * TicksPerSecond / Stopwatch.Frequency;

    long intPart = (long)ntpData[40] << 24 | (long)ntpData[41] << 16 | (long)ntpData[42] << 8 | ntpData[43];
    long fractPart = (long)ntpData[44] << 24 | (long)ntpData[45] << 16 | (long)ntpData[46] << 8 | ntpData[47];
    long netTicks = intPart * TicksPerSecond + (fractPart * TicksPerSecond >> 32);

    var networkDateTime = new DateTime(TicksTo1900 + netTicks + pingTicks / 2);

    return networkDateTime.ToLocalTime(); // without ToLocalTime() = faster
    }


    void requestNTPtime(){
            savingDisplay.gameObject.SetActive(false);
        try{
            print("New request");
            DateTime t = GetNetworkTime();
            requestTime = ((System.DateTime.UtcNow.Ticks - 621355968000000000) / 10000000.0d);
            startTime = ((t.Ticks - 621355968000000000) / 10000000.0d - 7200);
            stopGetTime = true;
            handContainer.gameObject.SetActive(true);
            savingDisplay.gameObject.SetActive(true);
        }
        catch{
            print("failed");
        }
    }


    double startTime;
    double requestTime;
    bool stopGetTime = false;
    void Start(){
        requestNTPtime();
    }



    double getTime(){
        double now = ((System.DateTime.UtcNow.Ticks - 621355968000000000) / 10000000.0d);
        return startTime + now - requestTime;
    }


    
    // Update is called once per frame (50 Hz)
    void FixedUpdate()
    {
        if(!stopGetTime){
            requestNTPtime();
        }


        getRemoteInputs();

        if (isSaving)
        {
            
            string frameLeftHand = computeFrameDesc(leftHand, leftHand.GetComponent<OVRHand>(), 0);
            string frameRightHand = computeFrameDesc(rightHand, rightHand.GetComponent<OVRHand>(), 4);
            string time = getTime().ToString();
            string gestureInfoRight = "None";
            string gestureInfoLeft = "None";
            string gestureShownRight = "None";
            string gestureShownLeft = "None";
            if(gestureDisplay.phase == 1)
            {
                gestureShownRight = gestureDisplay.imageToDisplay(gestureDisplay.imageCounter, 1).ToString();
                gestureShownLeft = gestureDisplay.imageToDisplay(gestureDisplay.imageCounter, 0).ToString();
            }

            if (gestureRecognizerRight.gestureDetected != null) gestureInfoRight = gestureRecognizerRight.gestureDetected.gestureName;
            if (gestureRecognizerLeft.gestureDetected != null) gestureInfoLeft = gestureRecognizerLeft.gestureDetected.gestureName;

            System.IO.File.AppendAllText(path, time + ";" + frameLeftHand  + gestureShownLeft + ";" + gestureInfoLeft + ";" + frameRightHand + gestureShownRight + ";" + gestureInfoRight + ";\n");
            
        }

        if (sendFeedBack) {
            System.IO.File.AppendAllText(Application.persistentDataPath + "/Data/startRecording.txt", " test");
            sendFeedBack = false;
        }
        
    }

    //tests if a file with a certain name has been created by a remote control to command an action
    private void getRemoteInputs()
    {
        string fileName;
        //test Start Recording
        fileName = Application.persistentDataPath + "/Data/startRecording.txt";
        if (System.IO.File.Exists(fileName) && !isSaving)
        {
            currentRecordingName = System.IO.File.ReadAllText(fileName);
            startRecording();
            sendFeedBack = true;
        }

        //test Stop Recording
        fileName = Application.persistentDataPath + "/Data/stopRecording.txt";
        if (System.IO.File.Exists(fileName) && isSaving)
        {
            System.IO.File.Delete(fileName);
            stopRecording();
        }
        
    }
    
    //starts the recording, creates a file to put the record in and displays that the recording has started
    private void startRecording()
    {
        isSaving = true;
        savingDisplay.GetComponent<Renderer>().material = savingMaterialOn; //display that the recording has started

        path = Application.persistentDataPath + "/Data/" + currentRecordingName + ".txt";
        System.IO.File.WriteAllText(path, "");
    }


    //stops the recording and displays that the recording has stopped
    private void stopRecording()
    {
        isSaving = false;
        savingDisplay.GetComponent<Renderer>().material = savingMaterialOff; //display that the recording has stopped
    }


   

    //gets all the informations on the hand gesture and formats it in a string on one line in csv format
    private string computeFrameDesc(OVRSkeleton handSkeleton, OVRHand hand, int fingerIndex)
    {
        OVRSkeleton.SkeletonPoseData pose = handSkeleton.getBoneData();
        
        string text = "";
        text += (pose.IsDataValid ? 1 : 0) + ";";
        //textDisplay.text = pose.RootPose.ToString();

        text += pose.RootPose.ToString() + ";"; 
        for (int i=0; i< 19; ++i)
        {
            Vector3 boneRotation = pose.BoneRotations[i].FromFlippedXQuatf().eulerAngles;
            text += (boneRotation.x + ";" + boneRotation.y + ";" + boneRotation.z).Replace(",", ".").Replace(";", ",") + ";";
        }
        for (int i = 1; i < 5; ++i)
        {
            bool pinching = hand.GetFingerIsPinching((OVRHand.HandFinger)i);
            text += (pinching ? 1 : 0) + ";";
        }
        for (int i = 0; i < 5; ++i)
        {
            text += (hand.GetFingerConfidence((OVRHand.HandFinger)i).ToString() == "High" ? 1 : 0) + "; ";
        }

        return text;
    }
}
