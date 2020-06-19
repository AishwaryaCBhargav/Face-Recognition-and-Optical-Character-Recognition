clc;
clear all;

path = fullfile('/Users/aishwarya/Documents/City MSc Data Science/Term 2/Computer Vision/Models/OCR');

dir;

files = dir(path);
Dirs = [files.isdir];
subdirectories = files(Dirs);

length(subdirectories)

for i = 1:length(subdirectories)
    
    subfolder=fullfile(path, subdirectories(i).name);
   
    files1 = dir(subfolder)
    dirFlags1 = [files1.isdir]&~strcmp({files1.name},'.')&~strcmp({files1.name},'..')&~strcmp({files1.name},'.DS_Store');
    
    % Get list of jpg and jpeg files in folder
    image_files_1=dir(fullfile(subfolder,'/I*.jp*'));
    image_files_2=dir(fullfile(subfolder,'/v*.jp*'));
    
    %%
    for j = 1 : length(image_files_1)
        image_files_name = fullfile(subfolder,image_files_1(j).name)
        I1=imread(image_files_name);        
        I1 = imresize(I1,0.4);
        IG1 = rgb2gray(I1);
        IGB1 = IG1>200;
        %imshow(IGB1);
        blobAnalyzer = vision.BlobAnalysis('MaximumCount', 500);

        % Run the blob analyser to find connected components
        [area, centroids, roi] = step(blobAnalyzer, IGB1); % ( OR CAN USE OCR FUNCTION TOO)


        for k = 1 : size(area,1)
            wordBBox = roi(k,:);
            % Show the location of the word in the original image
            if wordBBox(3)>150
                figure;
                hold
                Iname = insertObjectAnnotation(I1,'rectangle', wordBBox,k);
                imshow(Iname);

                wordBBoxInterest = wordBBox;
            end
        end

        box = I1(wordBBoxInterest(2):wordBBoxInterest(2)+wordBBoxInterest(4),wordBBoxInterest(1):wordBBoxInterest(1)+wordBBoxInterest(3));
        [sizex, sizey] = size(box);
        box = box(floor(sizex/4) : floor(sizex*3/4), floor(sizey/4):floor(sizey*3/4));
        box = imresize(box,0.5);
        resultsNew = ocr(box,'TextLayout','Block');
        text = deblank( {resultsNew.Text(1:2)} );
        img = insertObjectAnnotation(I1,'rectangle', wordBBoxInterest, text);
        figure;
        %imshowpair(I1, img,'montage');
    end
    %%    
                 
    for l = 1 : length(image_files_2)
        image_files_name = fullfile(subfolder,image_files_2(l).name)
        I2=imread(image_files_name);
        
        
        I2 = imresize(I2,1.4);
        IG2 = rgb2gray(I2);
        IGB2 = IG2>200;
        %imshow(IGB2);
        blobAnalyzer = vision.BlobAnalysis('MaximumCount', 500);

        % Run the blob analyser to find connected components
        [area, centroids, roi] = step(blobAnalyzer, IGB2); % ( OR CAN USE OCR FUNCTION TOO)


        for m = 1 : size(area,1)
            wordBBox = roi(m,:);
            % Show the location of the word in the original image
            if wordBBox(3)>150
                figure;
                hold
                Iname = insertObjectAnnotation(I2,'rectangle', wordBBox,m);
                imshow(Iname);

                wordBBoxInterest = wordBBox;
            end
        end
    
        box = I2(wordBBoxInterest(2):wordBBoxInterest(2)+wordBBoxInterest(4),wordBBoxInterest(1):wordBBoxInterest(1)+wordBBoxInterest(3));
        [sizex, sizey] = size(box);
        box = box(floor(sizex/4) : floor(sizex*3/4), floor(sizey/4):floor(sizey*3/4));
        box = imresize(box,1.5);
        resultsNew = ocr(box,'TextLayout','Block');
        text = deblank( {resultsNew.Text(1:2)} );
        img = insertObjectAnnotation(I2,'rectangle', wordBBoxInterest, text);
        figure;
        imshowpair(I2, img,'montage');
    end
    
end
