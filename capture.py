


    with mss.mss(display=':0.0') as sct:

        region={'top': 65, 'left': 64, 'width': 1020, 'height': 750}

        detector = TOD()

        while(True):

            last_time = time.time()

            print('Frame took {} seconds'.format(time.time()-last_time))

            screen = np.array(sct.grab(region))

            output_keys 	= keyboard_action.check_keys() #only one hot key:[1,0,0] or [0,1,0] or [0,0,1]

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            figure_vector 	= [detector.detect(screen),output_keys]

            print(figure_vector)

            training_data.append(figure_vector)



            if len(training_data) % 100 == 0:

                print(len(training_data))

                np.save(file_name, training_data)



            if cv2.waitKey(25) & 0xFF == ord('q'):

                cv2.destroyAllWindow()

                break

