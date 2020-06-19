% Defining a face recognition function
function [P] = RecogniseFace(I, featureType, classifierName)
FaceDetector = vision.CascadeObjectDetector('MergeThreshold', 8);

image = imread(I);
% steps through images, detecting faces
BBOX = step(FaceDetector,image);
image = insertObjectAnnotation(image,'rectangle',BBOX,'Face');
imshow(image),title('Detected Faces');

[r, ~] = size(BBOX);
P = [];

% load EmotionClassifier
load('EmotionClassifier.mat');

% if Hog + SVM
if strcmpi(featureType, 'Hog') && strcmpi(classifierName, 'SVM')
    % load classifier
    load('hog_svm_model.mat')
    % step through each face, and populate matrix P
    for x = 1:r
        face = image(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face1 = imresize(face, [90 90]);
        queryFeatures = extractHOGFeatures(face1);
        label = predict(faceClassifier, queryFeatures);
        label = string(label{1});
        
        %predicting emotion
        emotion = predict(emotionClassifier, queryFeatures);
        emotion = str2num(emotion{1});
        
        %finding the mid-point of the location of face
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P = [P;label, mid_x, mid_y, emotion];
        imshow(face);
    end


% if Bag + SVM
elseif strcmpi(featureType, 'Bag') && strcmpi(classifierName, 'SVM')
    % load classifier
    load('bag_svm_model.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = image(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);        
        face1 = imresize(face, [90 90]);
        queryFeatures = encode(bag,face1);
        label = predict(faceClassifier, queryFeatures);
        label = string(label{1});
        
        %predicting emotion
        queryFeatures1 = extractHOGFeatures(face1);
        emotion = predict(emotionClassifier, queryFeatures1);
        emotion = str2num(emotion{1});
        
        %finding the mid-point of the location of face
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P = [P;label, mid_x, mid_y, emotion]; %center point, %center point
        imshow(face);
    end

% if Bag + MLP
elseif strcmpi(featureType, 'Bag') && strcmpi(classifierName, 'MLP')
    % load classifier
    load('bag_fnn_model.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = image(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [90 90]);
        queryFeatures = encode(bag,face).';
        outPuts = net(queryFeatures);
        [value label] = max(outPuts(:,1));
        
        queryFeatures1 = extractHOGFeatures(face);
        emotion = predict(emotionClassifier, queryFeatures1);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P = [P;label, mid_x, mid_y, emotion]; %center point, %center point
        imshow(face);
    end     

% if HOG + MLP
elseif strcmpi(featureType, 'Hog') && strcmpi(classifierName, 'MLP')
    % load classifier
    load('hog_fnn_model.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = image(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [90 90]);
        queryFeatures = extractHOGFeatures(face);
        queryFeaturesT = queryFeatures.';
        outPuts = net(queryFeaturesT);
        [value label] = max(outPuts(:,1));
        
        queryFeatures = extractHOGFeatures(face);
        emotion = predict(emotionClassifier, queryFeatures);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P = [P;label, mid_x, mid_y, emotion]; %center point, %center point
        imshow(face);
    end
    
% if CNN
elseif strcmpi(featureType, 'CNN') && strcmpi(classifierName, 'CNN')
    %load classifier
    load('cnn_model.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = image(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3),:);
        face1 = imresize(face, [90 90]);
        imshow(face1);
        label = classify(net,face1);        
        label = string(label);
        
        queryFeatures1 = extractHOGFeatures(face1);
        emotion = predict(emotionClassifier, queryFeatures1);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P = [P;label, mid_x, mid_y, emotion]; %center point, %center point
        imshow(face);
    end
    
else
    % if the function is called without matching feature/classifier names,
    % the following error will show
    error('Please select from feature options ["HOG", "Bag", "CNN"] and classifier options ["SVM", "MLP", "CNN"])')
end
end
