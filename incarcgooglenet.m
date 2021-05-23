m = load ('googlenetthirdtryynewcode.mat')
loaded = m.Net

%img = readimage(imds,100);
%actualLabel = imds.Labels(100);

testImage = imread ('tulip.jpg')

resizedIm = imresize(testImage, [224 224]);

%[YPred,probs] = classify(m.Net,resizedIm)
predictedLabel = m.Net.classify(resizedIm)

imshow(resizedIm);
title(['Predicted: ' char(predictedLabel)])
