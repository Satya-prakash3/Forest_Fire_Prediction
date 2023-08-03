clc;
clear all;
close all;
warning off;

outputFolder=fullfile('D:\Deeplearning_work\Forest_FIRE\Forest Fire Dataset');
rootFolder=fullfile(outputFolder,'Training');
categories={'fire','nofire'};
imds=imageDatastore(fullfile(rootFolder,categories),'labelSource','foldernames');


%%
% To show the Samples
figure;
perm = randperm(1520,2);
for i = 1:2
    subplot(1,2,i);
    imshow(imds.Files{perm(i)});
end
%% To show the no. of samplesin each category
labelCount = countEachLabel(imds)
%% check the size of image
img = readimage(imds,1);
size(img)

%% specify the training set
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomize');


%% Define Network Architecture

layers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
%% specify training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'minibatchsize',64,...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'executionenvironment','gpu',...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');
%% Train network with training data
net = trainNetwork(imdsTrain,layers,options);


%% classify validation image and compute accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

figure;
confusionchart(YValidation,YPred);




