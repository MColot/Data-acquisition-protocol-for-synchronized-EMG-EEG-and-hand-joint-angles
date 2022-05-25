using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class customCharacterMovement : MonoBehaviour
{
    /*
     Allows to move the transform using inputs from the Oculus Quest Controllers
     - left joystick is used to move horizontally
     - left buttons are used to move vertically
     - right joystick is used to turn
         */
         
    void FixedUpdate()
    {   
        if (!OVRPlugin.GetHandTrackingEnabled())
        {
            Vector2 axis1 = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
            Vector2 axis2 = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);
            bool up = OVRInput.Get(OVRInput.Button.Four);
            bool down = OVRInput.Get(OVRInput.Button.Three);

            transform.position += new Vector3(axis1.x, 0, axis1.y) / 100f;
            if (up) transform.position += new Vector3(0, 0.001f, 0);
            if (down) transform.position += new Vector3(0, -0.001f, 0);
            transform.Rotate(new Vector3(0, axis2.x, 0) / 10f);
        }
    }
}
