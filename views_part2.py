        if len(faces) > 0 and frame_count % 15 == 0:

            print("Recognition Block Entered")

            try:

                x, y, w, h = faces[0]

                x1 = max(0, x)
                y1 = max(0, y)

                x2 = x1 + w
                y2 = y1 + h

                face_img = frame[y1:y2, x1:x2]

                face_img = cv2.resize(
                    face_img,
                    (224, 224)
                )

                if face_img.size > 0:

                    cv2.imwrite(
                        "media/current_face.jpg",
                        face_img
                    )

                    print("Recognition Started")

                    recognized_name = recognize_face(
                        "media/current_face.jpg"
                    )
                    emotion = detect_emotion(
                        "media/current_face.jpg"
                    )

                    blacklisted_person = check_blacklist(
                        "media/current_face.jpg"
                    )
                    
                    print("BLACKLIST RESULT =", blacklisted_person)

                    if blacklisted_person:

                        is_blacklisted = True

                        alert = Alert.objects.create(
                            alert_type="Blacklisted Person",
                            message=f"{blacklisted_person} detected in {camera_name}"
                        )

                        print("ALERT ID =", alert.id)

                        print(f"BLACKLIST DETECTED: {blacklisted_person}")

                        if can_send_email(
                            f"blacklist_{blacklisted_person}"
                        ):

                            send_alert_email(
                                "🚨 BLACKLISTED PERSON DETECTED",
                                f"{blacklisted_person} detected in {camera_name}"
                            )
                        filename = datetime.now().strftime(
                            "blacklist_%Y%m%d_%H%M%S.jpg"
                        )

                        path = os.path.join(
                            "media",
                            "blacklist_events",
                            filename
                        )

                        os.makedirs(
                            "media/blacklist_events",
                            exist_ok=True
                        )

                        cv2.imwrite(
                            path,
                            frame
                        )
                    # assign or reuse a numeric id for the recognized person
                    if recognized_name != "Unknown":
                        if recognized_name not in name_id_map:
                            name_id_map[recognized_name] = next_person_id
                            next_person_id += 1
                        current_person_id = name_id_map[recognized_name]
                    else:
                        # unknowns get a running U<number>
                        unknown_counter += 1
                        current_person_id = f"U{unknown_counter}"

                    print("Emotion =", emotion)

                    today = date.today()

                    # if recognized_name != "Unknown":

                    #     Alert.objects.create(
                    #         alert_type="Known Face",
                    #         message=f"{recognized_name} detected in {camera_name}"
                    #     )

                    #     if can_send_email(f"known_{recognized_name}"):

                    #         send_alert_email(
                    #             "Known Person Detected",
                    #             f"{recognized_name} detected in {camera_name}"
                    #         )

                    attendance_record = Attendance.objects.filter(
                        employee_name=recognized_name,
                        date=today
                    ).first()

                    if not attendance_record:
                        Attendance.objects.create(
                            employee_name=recognized_name,
                            status="Present"
                        )
                    else:
                        # Update the exit time on subsequent detections
                        attendance_record.exit_time = datetime.now().time()
                        attendance_record.save()

                    print("Recognized =", recognized_name)

                    now = datetime.now()
                    should_save_visitor = False
                    visitor_name = recognized_name
                    alert_type = None
                    alert_message = None

                    if recognized_name == "Unknown":
                        alert_type = "Unknown Face"
                        alert_message = "Unknown person detected"
                        should_save_visitor = (
                            last_unknown_time is None or
                            now - last_unknown_time > timedelta(seconds=30)
                        )
                    else:
                        if (
                            last_recognized_name != recognized_name or
                            last_recognized_time is None or
                            now - last_recognized_time > timedelta(seconds=30)
                        ):
                            should_save_visitor = True

                    if should_save_visitor:

                        if alert_type is not None:

                            Alert.objects.create(
                                alert_type="Unknown Face",
                                message=f"Unknown person detected in {camera_name}"
                            )

                            print("Unknown Visitor Detected")

                            if can_send_email("unknown_face"):

                                send_alert_email(
                                    "Unknown Face Detected",
                                    f"Unknown person detected in {camera_name}"
                                )

                        else:

                            Alert.objects.create(
                                alert_type="Known Face",
                                message=f"{recognized_name} detected in {camera_name}"
                            )

                            print(f"Known Visitor Detected: {recognized_name}")

                            if can_send_email(f"known_{recognized_name}"):

                                send_alert_email(
                                    "Known Person Detected",
                                    f"{recognized_name} detected in {camera_name}"
                                )

                        filename = now.strftime(
                            "%Y%m%d_%H%M%S"
                        ) + ".jpg"

                        visitor_folder = os.path.join(
                            "media",
                            "visitors"
                        )

                        os.makedirs(
                            visitor_folder,
                            exist_ok=True
                        )

                        visitor_path = os.path.join(
                            visitor_folder,
                            filename
                        )

                        cv2.imwrite(
                            visitor_path,
                            frame
                        )

                        visitor = Visitorlogo.objects.create(
                        visitor_name=visitor_name,
                        Snapshot=f"visitors/{filename}",
                        camera_name=camera_name
                    )

                        print(
                            "Visitor Saved:",
                            visitor.id
                        )

                        if recognized_name == "Unknown":
                            last_unknown_time = now
                        else:
                            last_recognized_time = now
                            last_recognized_name = recognized_name

                        

            except Exception as e:

                print("Recognition Error:", e)

                recognized_name = "Unknown"
        # ==========================
        # DRAW FACE BOX
        # ==========================
        if is_blacklisted:

            cv2.putText(
                frame,
                "BLACKLISTED PERSON",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
        for (x, y, w, h) in faces:

            if is_blacklisted:

                box_color = (0, 0, 255)
                text_color = (0, 0, 255)

            elif recognized_name != "Unknown":

                box_color = (0, 255, 0)
                text_color = (0, 255, 0)

            else:

                box_color = (0, 165, 255)
                text_color = (0, 165, 255)


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # Display name and assigned ID
            id_text = f"ID: {current_person_id}" if current_person_id is not None else "ID: -"
            cv2.putText(
                frame,
                f"{recognized_name} | {id_text}",
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )

            # Emotion
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # ==========================
        # STATS
        # ==========================

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # cv2.putText(
        #     frame,
        #     f"Name: {recognized_name}",
        #     (10, 70),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 0),
        #     2
        # )

        # cv2.putText(
        #     frame,
        #     f"Emotion: {emotion}",
        #     (10, 150),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (255, 0, 255),
        #     2
        # )

        cv2.putText(
            frame,
            phone_text,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ==========================
        # JPEG ENCODE
        # ==========================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 40]
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )
      

def video_feed(request, camera_id):
    try:
        camera = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        return HttpResponse("Camera not found", status=404)
        
    cam_key = f"camera_{camera.id}"
    
    return StreamingHttpResponse(
        generate_frames(camera.id, camera.name, cam_key),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )