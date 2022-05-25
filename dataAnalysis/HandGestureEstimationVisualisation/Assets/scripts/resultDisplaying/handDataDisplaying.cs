using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class handDataDisplaying : MonoBehaviour
{
    /*
     Loads motion capture data from a csv file and displays it on a OVR hand (can be real motion capture or estimation of motion capture from a model)
         */

    public Transform[] reconstructedHand;
    public string filePath;

    private StreamReader reader;
    public int startRotationIndex = 3;
    private int endRotationIndex;

    public int slower = 0;
    private int currSlowing = 0;

    private List<List<Quaternion>> lastRotations = new List<List<Quaternion>>();
    public int smooth = 1;

    private void Start()
    {
        reader = new StreamReader(filePath);
        endRotationIndex = startRotationIndex + 17;
    }

    void loadFromRecordFile()
    {
        string line = "";
        List<Quaternion> boneRotations = new List<Quaternion>();
        int indexInLine = 0;
        string current = "";
        for (int i = 0; i < line.Length; ++i)
        {
            char c = line[i];
            if (c == ';')
            {
                if (indexInLine >= startRotationIndex && indexInLine < endRotationIndex)
                {
                    boneRotations.Add(Quaternion.Euler(stringToVector3(current)));
                }
                ++indexInLine;
                current = "";
            }
            else
            {
                current += c;
            }
        }

        lastRotations.Add(boneRotations);
        if(lastRotations.Count > smooth){
            lastRotations.RemoveAt(0);
        }

        for (int i = 0; i < reconstructedHand.Length; ++i)
        {
            Quaternion mean = new Quaternion(0, 0, 0, 0);
            for(int j=0 ; j<lastRotations.Count ; ++j){
                Quaternion q = lastRotations[j][i];
                int c = lastRotations.Count;
                mean = new Quaternion(mean.x + q.x/c, mean.y + q.y/c, mean.z + q.z/c, mean.w + q.w/c);
            }
            //reconstructedHand[i].localRotation = boneRotations[i];
            reconstructedHand[i].localRotation = mean;
        }
    }




    // Update is called once per frame
    void FixedUpdate()
    {
        if(currSlowing < slower)
        {
            ++currSlowing;
            return;
        }
        currSlowing = 0;


        string line = reader.ReadLine();
        if (line is null)
        {
            reader = new StreamReader(filePath);
        }
        else
        {
            List<Quaternion> boneRotations = new List<Quaternion>();
            int indexInLine = 0;
            string current = "";
            for (int i = 0; i < line.Length; ++i)
            {
                char c = line[i];
                if (c == ';')
                {
                    if (indexInLine >= startRotationIndex && indexInLine < endRotationIndex)
                    {
                        boneRotations.Add(Quaternion.Euler(stringToVector3(current)));
                    }
                    ++indexInLine;
                    current = "";
                }
                else
                {
                    current += c;
                }
            }

            lastRotations.Add(boneRotations);
            while (lastRotations.Count > smooth)
            {
                lastRotations.RemoveAt(0);
            }

            for (int i = 0; i < reconstructedHand.Length; ++i)
            {
                Quaternion mean = new Quaternion(0, 0, 0, 0);
                for (int j = 0; j < lastRotations.Count; ++j)
                {
                    Quaternion q = lastRotations[j][i];
                    int c = lastRotations.Count;
                    mean = new Quaternion(mean.x + q.x / c, mean.y + q.y / c, mean.z + q.z / c, mean.w + q.w / c);
                }
                reconstructedHand[i].localRotation = mean;
            }
            

        }
    }


    private Vector3 stringToVector3(string s)
    {
        s += ",";
        float[] values = new float[3];
        int index = 0;
        string current = "";
        for (int i = 0; i < s.Length; ++i)
        {
            char c = s[i];
            if (c == ',')
            {
                values[index] = float.Parse(current);
                current = "";
                ++index;
            }
            else if(c == '.')
            {
                current += ',';
            }
            else
            {
                current += c;
            }
        }

        return new Vector3(values[0], values[1], values[2]);
    }
}
