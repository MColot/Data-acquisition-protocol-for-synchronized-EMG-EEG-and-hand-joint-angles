using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class gestureDisplay : MonoBehaviour
{
    /*
     Handles the display of the current exercice to the user
     - waits for remote command that tells what exercice to display
     - updates the displays (free move exercices and signs gestures to perform)
         */

    public int phase = 0;

    public GameObject handMvcDisplay;
    public GameObject signLanguageDisplay;
    public GameObject pinchingDisplay;
    public GameObject freeMoveDisplay;
    public GameObject utcDisplay;

    public GameObject[] freemoveTables;
    public Transform[] ovrHandRenderer;
    public Transform[] autoHandRenderer;
    
    public List<Sprite> signLanguageImages;
    public Sprite signLanguageNeutralImage;
    public GameObject pronationDisplay;
    public GameObject supinationDisplay;
    public TextMesh descriptionDisplay;


    public SpriteRenderer signLanguageSpriteRendererLeft;
    public SpriteRenderer signLanguageSpriteRendererRight;
    public GameObject goodResultDisplayLeft;
    public GameObject goodResultDisplayRight;

    private int[] currentImageId = new int[] { -1, -1 };
    public int imageCounter = -1;
    private bool waitingForGesture = true;

    private int repetitions = 0;
    private List<int> gesturesToPerform;

    private int neutralPose = 12;
    private int curHandToMoveSign = 0;
    private int stepInSign = 0;

    

    private void Update()
    {
        //display MVC
        string fileName = Application.persistentDataPath + "/Data/mvc.txt";
        if (System.IO.File.Exists(fileName))
        {
            System.IO.File.Delete(fileName);
            displayMVC();
        }

        //display sign language
        fileName = Application.persistentDataPath + "/Data/signLang.txt";
        if (System.IO.File.Exists(fileName))
        {
            string param = System.IO.File.ReadAllText(fileName);
            parseParamSignLang(param);
            System.IO.File.Delete(fileName);
            displaysignLanguage();
        }

        //display free moves
        fileName = Application.persistentDataPath + "/Data/freemove.txt";
        if (System.IO.File.Exists(fileName))
        {
            string param = System.IO.File.ReadAllText(fileName);
            System.IO.File.Delete(fileName);
            displayFreeMove(param);
        }

        //display utc
        fileName = Application.persistentDataPath + "/Data/utc.txt";
        if (System.IO.File.Exists(fileName))
        {
            System.IO.File.Delete(fileName);
            displayUTC();
        }
    }

    private void displayUTC(){
        phase = 2;
        freeMoveDisplay.SetActive(false);
        signLanguageDisplay.SetActive(false);
        handMvcDisplay.SetActive(false);
        utcDisplay.SetActive(true);
    }

    private void displayMVC()
    {
        phase = 0;
        freeMoveDisplay.SetActive(false);
        signLanguageDisplay.SetActive(false);
        handMvcDisplay.SetActive(true);
        utcDisplay.SetActive(false);

        for (int i = 0; i < ovrHandRenderer.Length; ++i) ovrHandRenderer[i].gameObject.layer = 0;
        for (int i = 0; i < autoHandRenderer.Length; ++i) autoHandRenderer[i].gameObject.SetActive(false);
    }

    private void displaysignLanguage()
    {
        phase = 1;
        freeMoveDisplay.SetActive(false);
        signLanguageDisplay.SetActive(true);
        handMvcDisplay.SetActive(false);
        utcDisplay.SetActive(false);
        goodResultDisplayLeft.SetActive(false);
        goodResultDisplayRight.SetActive(false);
        curHandToMoveSign = 0;
        stepInSign = 0;
        gestureFound = false;
        updateImage();

        for (int i = 0; i < ovrHandRenderer.Length; ++i) ovrHandRenderer[i].gameObject.layer = 0;
        for (int i = 0; i < autoHandRenderer.Length; ++i) autoHandRenderer[i].gameObject.SetActive(false);
    }

    private void displayFreeMove(string param)
    {
        phase = 2;
        freeMoveDisplay.SetActive(true);
        signLanguageDisplay.SetActive(false);
        handMvcDisplay.SetActive(false);
        utcDisplay.SetActive(false);

        int tableToDisplay = int.Parse(param);
        for(int i=0; i<freemoveTables.Length; ++i)
        {
            freemoveTables[i].SetActive(i == tableToDisplay);
        }

        for (int i = 0; i < ovrHandRenderer.Length; ++i)
        {
            if(tableToDisplay == 0) ovrHandRenderer[i].gameObject.layer = 0;
            else ovrHandRenderer[i].gameObject.layer = 12;
        }
        for (int i = 0; i < autoHandRenderer.Length; ++i) autoHandRenderer[i].gameObject.SetActive(tableToDisplay != 0);
    }


    bool gestureFound = false;
    public void notifyGesture(int imageId, int handId)
    {
        if (phase != 1 || !waitingForGesture) return;
        if (handId != curHandToMoveSign) return;
        if(imageId == currentImageId[handId] && !gestureFound)
        {
            goodResultDisplayLeft.SetActive(true);
            goodResultDisplayRight.SetActive(true);
            //Invoke("displayNeutral", 1);
            gestureFound = true;
            Invoke("updateImage", 2);
        }
    }

    public void displayNeutral()
    {
        signLanguageSpriteRendererLeft.sprite = signLanguageNeutralImage;
        signLanguageSpriteRendererRight.sprite = signLanguageNeutralImage;
    }

    public void updateImage()
    {
        if (stepInSign == 0 && curHandToMoveSign == 0) ++imageCounter;

        gestureFound = false;
        ++stepInSign;
        if (stepInSign == 2)
        {
            stepInSign = 0;
            if (curHandToMoveSign == 1) curHandToMoveSign = 0;
            else curHandToMoveSign = 1;
        }


        if(curHandToMoveSign == 0) goodResultDisplayLeft.SetActive(false);
        if (curHandToMoveSign == 1) goodResultDisplayRight.SetActive(false);

        int nextImageIdLeft = imageToDisplay(imageCounter, 0);
        int nextImageIdRight = imageToDisplay(imageCounter, 1);

        if (nextImageIdLeft >= signLanguageImages.Count) return;
        if (nextImageIdRight >= signLanguageImages.Count) return;
        if (imageCounter >= repetitions * gesturesToPerform.Count) return;

        currentImageId[0] = nextImageIdLeft;
        currentImageId[1] = nextImageIdRight;

        signLanguageSpriteRendererLeft.sprite = signLanguageImages[nextImageIdLeft];
        signLanguageSpriteRendererRight.sprite = signLanguageImages[nextImageIdRight];
        waitingForGesture = true;

        
    }

    public int imageToDisplay(int counter, int hand)
    {
        if (hand != curHandToMoveSign || stepInSign == 1) return neutralPose;

        int imageId = gesturesToPerform[counter % gesturesToPerform.Count];

        descriptionDisplay.text = "Repetition " +  (1+ (counter / gesturesToPerform.Count)).ToString() + "/" + repetitions.ToString() + ", gesture " + imageId.ToString();
        
        return imageId;
    }


    void parseParamSignLang(string p)
    {
        currentImageId[0] = -1;
        currentImageId[1] = -1;
        imageCounter = -1;
        repetitions = 0;
        gesturesToPerform = new List<int>();

        string curr = "";
        char c = p[0];
        int i = 0;
        while(c != '\n')
        {
            curr += c;
            ++i;
            c = p[i];
        }
        repetitions = int.Parse(curr);
        curr = "";
        while(i < p.Length-1)
        {
            ++i;
            c = p[i];
            if (c != ' ') curr += c;
            else
            {
                gesturesToPerform.Add(int.Parse(curr));
                curr = "";
            }
        }
        gesturesToPerform.Add(int.Parse(curr));
    }
}
