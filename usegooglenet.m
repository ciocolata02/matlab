outputFolder = fullfile('D:\Downloads');
rootFolder = fullfile (outputFolder, '15k', '15k'); %we use ; so the line is not executed everytime we run the program

categories = {'Acianthera Luteola'
'African Daisy'
'Anthurium'
'Buddleja'
'Cherry Blossom'
'Colorado four oclock'
'Common Dandelion'
'Daisy'
'Dames Rocket'
'Daphnes'
'Dayflowers'
'Duranta'
'Euonymus Fortunei'
'Evergreen Rose'
'Field Marigold'
'Floribunda'
'Hawaiian hibiscus'
'Hollyhocks'
'Hyacinth'
'Hypericum'
'Lily'
'Lily of the Nile'
'Lobelia'
'Lobster-Claws'
'Lupine'
'Mexican Marigold'
'Not a flower'
'Nyctaginaceae'
'Oleander'
'Oxeye Daisy'
'Phlox'
'Poinsettia'
'Primrose'
'Purple Coneflower'
'Redwood Sorrel'
'Rosa Centifolia Muscosa'
'Rose'
'Sacred Lotus'
'Sasanqua Camellia'
'Shepherdia'
'Sunflower'
'Sweet Grass'
'Tricyrtis'
'Tulip'
'Violet'
'Waling-Waling'
'Water Lilies'
'Wild Peony'
'Woodland Sunflower'}


imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');

tbl = countEachLabel(imds)

 Net = googlenet();
 
[trainingSet, testSet] = splitEachLabel (imds, 0.8, 'randomize');
 
numClasses = numel(categories(trainingSet.Labels));
lgraph = layerGraph(Net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'loss3-classifier',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

inputSize = Net.Layers(1).InputSize;
inputSize= [224 224];
augimdsTrain = augmentedImageDatastore(inputSize,trainingSet);
augimdsValidation = augmentedImageDatastore(inputSize,testSet);

numel(Net.Layers(end).ClassNames)

w1 = Net.Layers(2).Weights; %this line gets the weights of the second layer and stores them in the matrix w1
w1 = mat2gray(w1); %to visualize the matrix, we need to convert it to image


featureLayer = 'loss3-classifier';
trainingFeatures = activations(Net, augimdsTrain, featureLayer, 'MiniBatchSize', 32,'OutputAs', 'columns'); %minibatchsize is set to 32 to ensure that the CNN and the image data into GPU memory; if the GPU runs out of memory we need to lower the MinniBatchSize

trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learner','Linear','Coding','onevsall','ObservationsIn','columns');%fitcecoc = Fit-Class Error Correcting Output Codes; it returns a full trained multiclass error correcting output coded model;


testFeatures = activations(Net,augimdsValidation, featureLayer, 'MiniBatchSize',32,'OutputAs', 'columns'); %minibatchsize is set to 32 to ensure that the CNN and the image data into GPU memory; if the GPU runs out of memory we need to lower the MinniBatchSize

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns'); %returns a vector of predicted class levels based on trained classifier

testLabels = testSet.Labels;   %in this variable we have the actual levels and the predicted labels are in the predictLabels variable
confMat = confusionmat(testLabels, predictLabels);%we use this to evaluate the performance of the classifier

confMat = bsxfun(@rdivide, confMat, sum(confMat,2));%we want to divide the confusion matrix by a sinble column matrix
%sum(confMat,2) - calculatez the sum of the entire row of confMat and put
%the value on the first element

mean(diag(confMat)); % this shows the accuracy of the training set: 0.8545



%Now we test the image:
newImage = imread(fullfile('test102.jpg'));

resizedImage = imresize (newImage, [224 224])

imageFeatures = activations(Net, resizedImage, featureLayer, 'MiniBatchSize', 32,'OutputAs', 'columns');

label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

sprintf('The loaded image belongs to %s class', label)

imshow(resizedImage)
title(['Predicted: ' char(label)])

idx = randperm(numel(testSet.Files),4);
figure
for i = 1:4
 subplot(2,2,i)
 I = readimage(testSet,idx(i));
 imshow(I)
 label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')
 title(string(label));
end
