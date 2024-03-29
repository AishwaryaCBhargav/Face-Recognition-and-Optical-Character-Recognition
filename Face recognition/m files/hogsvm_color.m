
clc;
clear all;

tic;
path = fullfile('/Users/aishwarya/Documents/City MSc Data Science/Term 2/Computer Vision/Models/color_models/Models_27th');
imgFolder = fullfile(path, 'hog_svm');

faceDatabase = imageSet(imgFolder, 'recursive');


%Split Database into Training & Test Sets
rng(9);
[training,test] = partition(faceDatabase,[0.8 0.2], 'random');

%Extract HOG Features for Training Set

trainingSets = numel(training)  ;       % number of categories (69)
trainingSetSize = sum([training.Count]); % total number of training images

featureCount = 1;


% loop through each image, extract and add HOG Features to trainingFeatures
% matrix, and create vector holding label data per image
for i=1:trainingSets
    label = training(i).Description;
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = label;
        featureCount = featureCount + 1;
    end
end

%Train SVM using extracted HOG features and class labels 1-81
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%Extract HOG Features for Test Set

testSets = numel(test)  ;       % number of categories (69)
testSetSize = sum([test.Count]);        % total number of test images
testFeatureCount = 1;

% loop through each image in the test set, extract the HOG Features and the
% labels
for i=1:testSets
    actualLabel = test(i).Description;
    for j=1:test(i).Count
        testFeatures(testFeatureCount,:) = extractHOGFeatures(read(test(i),j));
        actualTestLabels{testFeatureCount, :} = actualLabel;
        testFeatureCount = testFeatureCount + 1;
    end
end

%Predict matching labels for all images in the test set

testLabels = predict(faceClassifier, testFeatures);

%Calculate accuracy of test imageset

correctMatches = 0;

% Check whether predicted label matches actual label for each image, and
% count how many are equivalent
for i=1:testSetSize
    if strcmp(testLabels{i}, actualTestLabels(i,:))
       correctMatches = correctMatches + 1;
    end
end

% calculate accuracy
accuracy = correctMatches/testSetSize

%Confusion matrix and average accuracy
ConfusiomMatrix = confusionmat(actualTestLabels, testLabels)
avg_accuracy = 100*sum(diag(ConfusiomMatrix))/sum(ConfusiomMatrix(:))
toc;

save hog_svm_model.mat