# Holistic Tracker

This module predicts pose + left/right hand + face landmarks.
It use the pose landmark as basic, then choosing the specific landmarks from the hand or head.

- Step 1: do the pose detection
- Step 2: do the pose landmark
- Step 3: Do the Handlandmark based on poselandmark
  - Step 3.1: Extract left-hand related landmarks.
  - Step 3.2: check if the hand is visibility. Actually,  we check if the poose wrist is visible. If not, it will prevent from preding hand landmarks on current frame.
  - Step 3.3: Get hand Roi based on left-hand related landmarks
  - Step 3.4: Predict hand re-crop rectangle on current frame by hand re-crop model.
  - Step 3.5: Get hand tracking rectangle for smoothing the final rectangle so that can get stable hand landmark.
  - Step 3.6: Get hand landmark by hand lanmark model.
  - Step 3.7 do the same to the right hand
- Step 4: do face landmark.
  - Step 4.1: Extract face related landmarks.
  - Step 4.2: Get face roi base on pose landmarks.
  - Step 4.3: refined face roi based on face detection model.
  - Step 4.4: Get face tracking roi.
  - Step 4.5: Get face landmarks.

