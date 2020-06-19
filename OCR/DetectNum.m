function [Q] = DetectNum(I)

image = imread(I);
I1 = imresize(image,[720 720]);
IG = rgb2gray(I1);
IGB = IG>200;
blobAnalyzer = vision.BlobAnalysis('MaximumCount', 500);

% Run the blob analyser to find connected components
[area, centroids, roi] = step(blobAnalyzer, IGB);

for k = 1 : size(area,1)
    wordBBox = roi(k,:);
    % Show the location of the word in the original image
    if wordBBox(3)>150
        figure;
        hold
        Iname = insertObjectAnnotation(I1,'rectangle', wordBBox,k);
        wordBBoxInterest = wordBBox;
    end
end

box = I1(wordBBoxInterest(2):wordBBoxInterest(2)+wordBBoxInterest(4),wordBBoxInterest(1):wordBBoxInterest(1)+wordBBoxInterest(3));
[sizex, sizey] = size(box);
box = box(floor(sizex/4) : floor(sizex*3/4), floor(sizey/4):floor(sizey*3/4));
box = imresize(box,0.5);
resultsNew = ocr(box,'TextLayout','Block');
text = deblank( {resultsNew.Text(1:2)} )
img = insertObjectAnnotation(I1,'rectangle', wordBBoxInterest, text);
figure;
imshowpair(I1, img,'montage');
end

