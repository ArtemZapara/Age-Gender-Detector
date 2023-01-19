import cv2

def get_face_box(net, frame, conf_threshold=0.7):

    frameCopy = frame.copy()
    width = frameCopy.shape[1]
    height = frameCopy.shape[0]

    blob = cv2.dnn.blobFromImage(
        image=frameCopy,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 117, 123),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:

            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            # print(x1, y1, x2, y2)

            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameCopy, (x1, y1), (x2, y2), (0, 255, 0), int(round(height/150)), 8)

    return frameCopy, boxes

def age_gender_detector(faceNet, ageNet, genderNet, image, pad=20, conf_thredhold = 0.3):

    # frameResized = cv2.resize(image, (640, 480))
    frameResized = image.copy()
    frame = frameResized.copy()

    frameCopy, boxes = get_face_box(faceNet, frame)

    ageList = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    genderList = ["Male", "Female"]
    for box in boxes:
        face = frame[max(0, box[1]-pad) : min(box[3]+pad, frame.shape[0]-1), \
                    max(0, box[0]-pad) : min(box[2]+pad, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(
            image = face,
            scalefactor = 1.0,
            size = (227, 227),
            mean = (78.4263377603, 87.7689143744, 114.895847746),
            swapRB = False
        )

        age = ""
        ageNet.setInput(blob)
        agePredictions = ageNet.forward()
        conf = agePredictions[0].max()
        if conf > conf_thredhold:
            age = ageList[agePredictions[0].argmax()]
        print(f"Predicted age: {age}, conf = {conf:.3f}")

        gender = ""
        genderNet.setInput(blob)
        genderPredictions = genderNet.forward()
        conf = genderPredictions[0].max()
        if conf > conf_thredhold:
            gender = genderList[genderPredictions[0].argmax()]
        print(f"Predicted gender: {gender}, conf = {conf:.3f}")

        label = f"{gender}{age}"
        cv2.putText(frameCopy, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    return frameCopy