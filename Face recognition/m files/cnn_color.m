clc;
clear all;

tic;

digitDatasetPath = fullfile('/Users/aishwarya/Documents/City MSc Data Science/Term 2/Computer Vision/Models/color_models/Models_27th/cnn');
faceDatabase = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');

%Counting the labels
labelCount = countEachLabel(faceDatabase);

img = readimage(faceDatabase,1);
size(img);

% Split Database into Training & Test Sets
numTrainFiles = 11;
[faceDatabaseTrain,faceDatabaseValidation] = splitEachLabel(faceDatabase,numTrainFiles,'randomize');

%model
layers = [
    imageInputLayer([90 90 3])
    
    convolution2dLayer(6,12,'Padding','same')  %size and no. of filters
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(6,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(6,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
       
    fullyConnectedLayer(69) %number of classes
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',faceDatabaseValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(faceDatabaseTrain,layers,options);

YPred = classify(net,faceDatabaseValidation);
YValidation = faceDatabaseValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

toc;

save cnn_model.mat