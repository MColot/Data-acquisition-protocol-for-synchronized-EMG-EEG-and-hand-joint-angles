using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using System;

public class multipleGestureRecognizer : MonoBehaviour
{
    /*
     Handles the recognition of the predefined gestures for one hand
     Sends triggers when a gesture is recognized
         */

    public GestureRecognizer gestureRecognizer;
    public OVRSkeleton rightHand;
    public TextMesh textDisplay;
    public gestureDisplay display;

    public TextMesh textDisplay2;
    public int handId = 0;

    public bool invertAxis = false;

    private void Start()
    {
        IList<OVRBone> bonesR = rightHand.Bones;
        gestureRecognizer.hand = bonesR[0].Transform.gameObject;
        for (int i = 0; i < 23; ++i)
        {
            gestureRecognizer.fingers[i] = bonesR[i+1].Transform.gameObject;
        }

        //test import gestures from file at start
        string fileName = Application.persistentDataPath + "/Data/importGesturesStart.txt";
        if (System.IO.File.Exists(fileName))
        {
            importGestures(fileName);
        }
        
    }


    void Update()
    {
        //test save gesture
        string fileName = Application.persistentDataPath + "/Data/saveGesture.txt";
        if (System.IO.File.Exists(fileName))
        {
            //textDisplay.text += "Saving new Gesture \n";
            System.IO.File.Delete(fileName);
            saveCurrentGesture();
        }

        //test import gestures from file
        fileName = Application.persistentDataPath + "/Data/importGestures.txt";
        if (System.IO.File.Exists(fileName))
        {
            //textDisplay.text += "importing Gestures \n";
            importGestures(fileName);
            System.IO.File.Delete(fileName);
        }

        //test export gestures in file
        fileName = Application.persistentDataPath + "/Data/exportGestures.txt";
        if (System.IO.File.Exists(fileName))
        {
            //textDisplay.text += "Exporting Gestures \n";
            System.IO.File.Delete(fileName);
            exportGestures();
        }
        
    }

    public void saveCurrentGesture()
    {
        try
        {
            gestureRecognizer.SaveAsGestureWithName((gestureRecognizer.savedGestures.Count).ToString());
            gestureRecognizer.savedGestures[gestureRecognizer.savedGestures.Count - 1].onRecognized.AddListener(() => { notifyGesture(); });

            //textDisplay.text += "New gesture saved \n";
        }
        catch(Exception e)
        {
            //textDisplay.text += e + "\n";
        }
    }

    public void notifyGesture()
    {
        textDisplay.text = "Gesture detected : " + gestureRecognizer.gestureDetected.gestureName;

        int gesture;
        bool b = int.TryParse(gestureRecognizer.gestureDetected.gestureName, out gesture);
        display.notifyGesture(gesture, handId);
    }

    
    public void onNothingDetected()
    {
        textDisplay.text = "No gesture detected";
    }


    private void exportGestures()
    {
        string path = Application.persistentDataPath + "/Data/GesturesExport.txt";
        foreach (Gesture g in gestureRecognizer.savedGestures)
        {
            System.IO.File.AppendAllText(path, g.gestureName + ":");
            foreach (Vector3 pos in g.positionsPerFinger)
            {
                System.IO.File.AppendAllText(path, pos.ToString("F8") + ";");
            }
            System.IO.File.AppendAllText(path, "$\n");
        }
    }

    private void importGestures(string fileName)
    {
        string fileContent = System.IO.File.ReadAllText(fileName);

        string currentString = "";
        Gesture currentGesture = new Gesture("", new List<Vector3>());
        foreach(char c in fileContent)
        {
            if(c == ':')
            {
                //textDisplay.text += "New gesture imported : " + currentString + " \n";
                currentGesture.gestureName = currentString;
                currentString = "";
            }
            else if(c == ';')
            {
                //textDisplay.text += "new value : " + currentString + " \n";
                currentGesture.positionsPerFinger.Add(parseVector3(currentString));
                currentString = "";
            }
            else if(c == '$')
            {
                currentGesture.onRecognized.AddListener(() => { notifyGesture(); });
                gestureRecognizer.savedGestures.Add(currentGesture);
                //textDisplay.text += "importation done : "+ gestureRecognizer.savedGestures.Count +" gestures saved\n";
                currentGesture = new Gesture("", new List<Vector3>());

            }
            else if (c != '\n')
            {
                currentString += c;
            }
        }
    }

    private Vector3 parseVector3(string s)
    {
        Vector3 res = new Vector3(0, 0, 0);
        int id = 0;
        string current = "";
        foreach(char c in s)
        {
            if(c == ',' || c == ')')
            {
                float v;
                bool b = float.TryParse(current.Replace(".", ","), out v);
                if (id == 0) res.x = v;
                else if (id == 1) res.y = v;
                else if (id == 2) res.z = v;
                ++id;
                current = "";
            }
            else if(c != '(')
            {
                current += c;
            }
        }
        
        if(invertAxis) return new Vector3(-res.x, -res.y, -res.z);
        return res;
    }
}
