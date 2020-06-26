import Leap, sys, thread
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
from math import sqrt
from math import fabs
import time
from sklearn.externals import joblib
import numpy as np


class LeapMotionListener(Leap.Listener):
    """
    LeapMotionListener class.
    """
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpo', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def on_init(self, controller):
        """
        Initialize.

        Args:
        ---------
            controller
        Return:
        ---------
        """
        print("Initialized")

    def on_connect(self, controller):
        """
        On connect method.

        Args:
        ---------
            controller
        Return:
        ---------
        """
        print("Motion Sensor Connected")

        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)

    def on_disconnect(self, controller):
        """
        On disconnect method.

        Args:
        ---------
            controller
        Return:
        ---------
        """
        print("Motion Sensor Disconnect")

    def on_exit(self, controller):
        """
        On exit method.

        Args:
        ---------
            controller
        Return:
        ---------
        """
        print("Exit")

    def on_frame(self, controller):
        """
        On frame method: get all features.

        Args:
        ---------
            controller
        Return:
        ---------
        """
        frame = controller.frame()
        # Get general properties
        # print("Frame ID: " + str(frame.id) + " Timestamp: " + str(frame.timestamp) + " # of hands: " + (len(frame.hands)) + " # of Fingers: " + str(len(frame.fingers)) + " # of Tools: " (len(frame.tools)) + " # of Gesture: " + (len(frame.gestures())))
        # print("Frame ID: " + str(frame.id)+ " Timestamp: " + str(frame.timestamp) + " # of hands: " + str(len(frame.hands)) + " # of Fingers: " + str(len(frame.fingers)) + " # of Gesture: " + str(len(frame.gestures())))

        DpalmtipThumb = 0
        DpalmtipIndex = 0
        DpalmtipMiddle = 0
        DpalmtipRing = 0
        DpalmtipPinky = 0
        DpalmtipTotal = 0

        ThumbVector = [0, 0, 0]
        IndexVector = [0, 0, 0]
        MiddleVector = [0, 0, 0]
        RingVector = [0, 0, 0]
        PinkyVector = [0, 0, 0]

        boneThumb = [0, 0, 0]
        boneIndex = [0, 0, 0]
        boneMiddle = [0, 0, 0]
        boneRing = [0, 0, 0]
        bonePinky = [0, 0, 0]

        boneTMetacarpal = [0, 0, 0]
        boneTProximal = [0, 0, 0]
        boneTIntermediate = [0, 0, 0]

        DTJ_ThumbJ3 = 0
        DTJ_IndexJ3 = 0
        DTJ_MiddleJ3 = 0
        DTJ_RingJ3 = 0
        DTJ_PinkyJ3 = 0

        DTJ_ThumbJ2 = 0
        DTJ_IndexJ2 = 0
        DTJ_MiddleJ2 = 0
        DTJ_RingJ2 = 0
        DTJ_PinkyJ2 = 0

        DTJ_ThumbJ1 = 0
        DTJ_IndexJ1 = 0
        DTJ_MiddleJ1 = 0
        DTJ_RingJ1 = 0
        DTJ_PinkyJ1 = 0

        sizeMetacarpoThumb = 0
        sizeMetacarpoIndex = 0
        sizeMetacarpoMiddle = 0
        sizeMetacarpoRing = 0
        sizeMetacarpoPinky = 0

        sizeProximalThumb = 0
        sizeProximalIndex = 0
        sizeProximalMiddle = 0
        sizeProximalRing = 0
        sizeProximalPinky = 0

        sizeIntermediateThumb = 0
        sizeIntermediateIndex = 0
        sizeIntermediateMiddle = 0
        sizeIntermediateRing = 0
        sizeIntermediatePinky = 0

        DT0JJ = 0
        DT1JJ = 0
        DT2JJ = 0
        DT3JJ = 0
        DT4JJ = 0

        RTP_0 = 0
        RTP_1 = 0
        RTP_2 = 0
        RTP_3 = 0
        RTP_4 = 0
        DT_12 = 0

        i = 0

        # Flag that enable to storage other simbol.
        FlagDifSimbol = False

        # Threshold of storage initialize.
        th = 0

        # Get hand and palm hand position/angles
        for hand in frame.hands:

            # Get data from each arm.
            normal = hand.palm_normal
            th = fabs(normal[1])

            # Track 500 points if NormYposition>.95.
            if (th > 0.9) and (i < 500) and (FlagDifSimbol is False):

                # Get data for each individual finger and bone
                for finger in hand.fingers:

                    # Obtain RTP, RTT and RTJ.
                    # RTP
                    # 1. Get palm center position.
                    hand_center = hand.palm_position

                    # 2. Calculate distance between palm and tip(alfa).
                    if finger.type == 0:
                        DpalmtipThumb = sqrt((finger.direction[0]-hand_center[0])**2 + (finger.direction[1]-hand_center[1])**2 + (finger.direction[2]-hand_center[2])**2)
                        ThumbVector = finger.direction

                    if finger.type == 1:
                        DpalmtipIndex = sqrt((finger.direction[0]-hand_center[0])**2 + (finger.direction[1]-hand_center[1])**2 + (finger.direction[2]-hand_center[2])**2)
                        IndexVector = finger.direction

                    if finger.type == 2:
                        DpalmtipMiddle = sqrt((finger.direction[0]-hand_center[0])**2 + (finger.direction[1]-hand_center[1])**2 + (finger.direction[2]-hand_center[2])**2)
                        MiddleVector = finger.direction

                    if finger.type == 3:
                        DpalmtipRing = sqrt((finger.direction[0]-hand_center[0])**2 + (finger.direction[1]-hand_center[1])**2 + (finger.direction[2]-hand_center[2])**2)
                        RingVector = finger.direction

                    if finger.type == 4:
                        DpalmtipPinky = sqrt((finger.direction[0]-hand_center[0])**2 + (finger.direction[1]-hand_center[1])**2 + (finger.direction[2]-hand_center[2])**2)
                        PinkyVector = finger.direction

                    DpalmtipTotal = DpalmtipThumb + \
                        DpalmtipIndex + \
                        DpalmtipMiddle + \
                        DpalmtipRing + \
                        DpalmtipPinky

                    # Features 1.
                    RTP_0 = DpalmtipThumb/DpalmtipTotal
                    RTP_1 = DpalmtipIndex/DpalmtipTotal
                    RTP_2 = DpalmtipMiddle/DpalmtipTotal
                    RTP_3 = DpalmtipRing/DpalmtipTotal
                    RTP_4 = DpalmtipPinky/DpalmtipTotal

                    # RTT
                    # 1. Distances between each Tip finger.
                    DT_12 = sqrt((ThumbVector[0]-IndexVector[0])**2 + (ThumbVector[1] - IndexVector[1])**2 + (ThumbVector[2] - IndexVector[2])**2)
                    DT_13 = sqrt((ThumbVector[0]-MiddleVector[0])**2 + (ThumbVector[1] - MiddleVector[1])**2 + (ThumbVector[2] - MiddleVector[2])**2)
                    DT_14 = sqrt((ThumbVector[0]-RingVector[0])**2 + (ThumbVector[1] - RingVector[1])**2 + (ThumbVector[2] - RingVector[2])**2)
                    DT_15 = sqrt((ThumbVector[0]-PinkyVector[0])**2 + (ThumbVector[1] - PinkyVector[1])**2 + (ThumbVector[2] - PinkyVector[2])**2)

                    DT_23 = sqrt((IndexVector[0]-MiddleVector[0])**2 + (IndexVector[1] - MiddleVector[1])**2 + (IndexVector[2] - MiddleVector[2])**2)
                    DT_24 = sqrt((IndexVector[0]-RingVector[0])**2 + (IndexVector[1] - RingVector[1])**2 + (IndexVector[2] - RingVector[2])**2)
                    DT_25 = sqrt((IndexVector[0]-PinkyVector[0])**2 + (IndexVector[1] - PinkyVector[1])**2 + (IndexVector[2] - PinkyVector[2])**2)
                    DT_34 = sqrt((MiddleVector[0]-RingVector[0])**2 + (MiddleVector[1] - RingVector[1])**2 + (MiddleVector[2] - RingVector[2])**2)
                    DT_35 = sqrt((MiddleVector[0]-PinkyVector[0])**2 + (MiddleVector[1] - PinkyVector[1])**2 + (MiddleVector[2] - PinkyVector[2])**2)
                    DT_45 = sqrt((MiddleVector[0]-PinkyVector[0])**2 + (MiddleVector[1] - PinkyVector[1])**2 + (MiddleVector[2] - PinkyVector[2])**2)

                    DEN_RTT = DT_12 + DT_13 + DT_14 + DT_15 + DT_23 + DT_24 + DT_25 + DT_34 + DT_35 + DT_45

                    # Features 2
                    RTT_01 = DT_12/DEN_RTT
                    RTT_02 = DT_13/DEN_RTT
                    RTT_03 = DT_14/DEN_RTT
                    RTT_04 = DT_15/DEN_RTT
                    RTT_12 = DT_23/DEN_RTT
                    RTT_13 = DT_24/DEN_RTT
                    RTT_14 = DT_25/DEN_RTT
                    RTT_23 = DT_34/DEN_RTT
                    RTT_24 = DT_35/DEN_RTT
                    RTT_34 = DT_45/DEN_RTT

                    # RTJ
                    # Get positions/directions from each hand bone
                    for b in range(0, 4):
                        bone = finger.bone(b)
                        # Position 3: Metacarpo.
                        if (bone.type == 0):
                            # IDENTIFY EACH FINGER, BEFORE!
                            if finger.type == 0:
                                DTJ_ThumbJ3 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeMetacarpoThumb = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 1:
                                DTJ_IndexJ3 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeMetacarpoIndex = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 2:
                                DTJ_MiddleJ3 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeMetacarpoMiddle = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 3:
                                DTJ_RingJ3 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeMetacarpoRing = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 4:
                                DTJ_PinkyJ3 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeMetacarpoPinky = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)

                        # Position 2: Proximal.
                        if (bone.type == 1):
                            if finger.type == 0:
                                DTJ_ThumbJ2 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeProximalThumb = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 1:
                                DTJ_IndexJ2 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeProximalIndex = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 2:
                                DTJ_MiddleJ2 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeProximalMiddle = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 3:
                                DTJ_RingJ2 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeProximalRing = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 4:
                                DTJ_PinkyJ2 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)						
                                sizeProximalPinky = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)

                        # Position 1: Intermediate.
                        if (bone.type == 2):
                            if finger.type == 0:
                                DTJ_ThumbJ1 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeIntermediateThumb = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 1:
                                DTJ_IndexJ1 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeIntermediateIndex = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 2:
                                DTJ_MiddleJ1 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeIntermediateMiddle = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 3:
                                DTJ_RingJ1 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeIntermediateRing = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)
                            if finger.type == 4:
                                DTJ_PinkyJ1 = sqrt((finger.direction[0]-bone.direction[0])**2 + (finger.direction[1]-bone.direction[1])**2 + (finger.direction[2]-bone.direction[2])**2)
                                sizeIntermediatePinky = sqrt((bone.next_joint[0]-bone.prev_joint[0])**2 + (bone.next_joint[1]-bone.prev_joint[1])**2 + (bone.next_joint[2]-bone.prev_joint[2])**2)

                        # OBTAIN: Dthumb = DThumbJ1J2 = DIntermediate + DProximal.
                        DT0JJ = sizeProximalThumb + sizeIntermediateThumb
                        DT1JJ = sizeProximalIndex + sizeIntermediateIndex
                        DT2JJ = sizeProximalMiddle + sizeIntermediateMiddle
                        DT3JJ = sizeProximalRing + sizeIntermediateRing
                        DT4JJ = sizeProximalPinky + sizeIntermediatePinky

                    # Features 3.
                    RTJ_0 = DTJ_ThumbJ3/(DTJ_ThumbJ1 + DT0JJ)
                    if (DTJ_IndexJ1 + DT1JJ) != 0:
                        RTJ_1 = DTJ_IndexJ3/(DTJ_IndexJ1 + DT1JJ)
                    if (DTJ_MiddleJ1 + DT2JJ) != 0:
                        RTJ_2 = DTJ_MiddleJ3/(DTJ_MiddleJ1 + DT2JJ)
                    if (DTJ_RingJ1 + DT3JJ) != 0:
                        RTJ_3 = DTJ_RingJ3/(DTJ_RingJ1 + DT3JJ)
                    if (DTJ_PinkyJ1 + DT4JJ) != 0:
                        RTJ_4 = DTJ_PinkyJ3/(DTJ_PinkyJ1 + DT4JJ)

                # Load model, then classify simbol
                tic = time.clock()
                self.get_inference(RTP_0, RTP_1, RTP_2, RTP_3, RTP_4, RTT_01, RTT_02, RTT_03, RTT_04, RTT_12, RTT_13, RTT_14, RTT_23, RTT_24, RTT_34, RTJ_0)
                toc = time.clock()
                print("Inference Processing time: " + str(toc-tic))

    def get_inference(self, RTP_0, RTP_1, RTP_2, RTP_3, RTP_4, RTT_01, RTT_02, RTT_03, RTT_04, RTT_12, RTT_13, RTT_14, RTT_23, RTT_24, RTT_34, RTJ_0):
        """
        Load model, then classify simbol
        Args:
        ---------
            RTP_0, RTP_1,...: features
        Return:
        ---------
            just print regonized characters
        """
        # classifier = joblib.load('data/NearestCentroid.pkl')
        # classifier = joblib.load('data/SVM.pkl')
        classifier = joblib.load('data/ANN.pkl')
        # classifier = joblib.load('data/NearestCentroid.pkl')

        InputSamples = np.vstack((
            RTP_0, RTP_1, RTP_2, RTP_3, RTP_4, RTT_01, RTT_02, RTT_03, RTT_04, RTT_12, RTT_13, RTT_14, RTT_23, RTT_24, RTT_34, RTJ_0
        ))
        InputSamples = InputSamples.T
        predicted = classifier.predict(InputSamples)
        if predicted == 1:
            print("A")
        if predicted == 2:
            print("B")
        if predicted == 3:
            print("C")
        if predicted == 4:
            print("D")
        if predicted == 5:
            print("E")
        if predicted == 6:
            print("F")
        if predicted == 7:
            print("G")
        if predicted == 8:
            print("H")
        if predicted == 9:
            print("I")
        if predicted == 10:
            print("K")
        if predicted == 11:
            print("L")
        if predicted == 12:
            print("M")
        if predicted == 13:
            print("N")
        if predicted == 14:
            print("O")
        if predicted == 15:
            print("P")
        if predicted == 16:
            print("Q")
        if predicted == 17:
            print("R")
        if predicted == 18:
            print("S")
        if predicted == 19:
            print("T")
        if predicted == 20:
            print("U")
        if predicted == 21:
            print("V")
        if predicted == 22:
            print("W")
        if predicted == 23:
            print("X")
        if predicted == 24:
            print("Y")

    def get_raw_data(self, RTP_0, RTP_1, RTP_2, RTP_3, RTP_4, RTT_01, RTT_02, RTT_03, RTT_04, RTT_12, RTT_13, RTT_14, RTT_23, RTT_24, RTT_34, RTJ_0, RTJ_1, RTJ_2, RTJ_3, RTJ_4):
        """
        Method that saves raw data into .txt file.

        Args:
        ---------
            RTP_0, RTP_1,...: features
        Return:
        ---------
            raw data saved
         """
        # Get the number of frames storage
        TotalLines = 0
        with open("FeaturesTest.txt") as f:
            TotalLines = sum(1 for _ in f)
            print("TEMP SIZE::::: " + str(sum(1 for _ in f)))

        # Storage training data
        if TotalLines < 500:
            ## Write features in column format
            readfile = open("data/FeaturesTest.txt","r")
            temp = str(readfile.read())
            file = open("data/FeaturesTest.txt","w")
            file.write(temp)
            file.write('\n')
            file.write(str(round(RTP_0,5)))
            file.write('\t')
            file.write(str(round(RTP_1,5)))
            file.write('\t')
            file.write(str(round(RTP_2,5)))
            file.write('\t')
            file.write(str(round(RTP_3,5)))
            file.write('\t')
            file.write(str(round(RTP_4,5)))
            file.write('\t')
            file.write(str(round(RTT_01,5)))
            file.write('\t')
            file.write(str(round(RTT_02,5)))
            file.write('\t')
            file.write(str(round(RTT_03,5)))
            file.write('\t')
            file.write(str(round(RTT_04,5)))
            file.write('\t')
            file.write(str(round(RTT_12,5)))
            file.write('\t')
            file.write(str(round(RTT_13,5)))
            file.write('\t')
            file.write(str(round(RTT_14,5)))
            file.write('\t')
            file.write(str(round(RTT_23,5)))
            file.write('\t')
            file.write(str(round(RTT_24,5)))
            file.write('\t')
            file.write(str(round(RTT_34,5)))
            file.write('\t')
            file.write(str(round(RTJ_0,5)))
            file.write('\t')
            file.write(str(round(RTJ_1,5)))
            file.write('\t')
            file.write(str(round(RTJ_2,5)))
            file.write('\t')
            file.write(str(round(RTJ_3,5)))
            file.write('\t')
            file.write(str(round(RTJ_4,5)))
            #file.write('\n')

        else:
            print("Clean set of training data...")


def main():
    listener = LeapMotionListener()
    controller = Leap.Controller()

    controller.add_listener(listener)

    print("Press enter to quit")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
