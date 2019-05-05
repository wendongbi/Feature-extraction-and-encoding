function count = get_proposal(path, img_name)

source_img = imread(path);
gray_img = rgb2gray(source_img);
T = graythresh(gray_img);
bw_img = im2bw(gray_img,T);
reverse_img = ~ bw_img;
% find the connected regions
region = regionprops(reverse_img,'area','boundingbox');
areas = [region.Area];
rects = cat(1,region.BoundingBox);
% figure(1)
% imshow(source_img)
img = [];
count = 0;
for i = 1:size(rects,1)
    img = imcrop(source_img,rects(i,:));
    width = size(img, 2);
    length = size(img, 1);
    if(width * length >= 100)
        
        count = count + 1;
        save_path = strcat('./test/',img_name, '_proposal_', num2str(count), '.jpg');
        fprintf('%s\n', save_path);
        imwrite(img, save_path);
    end
end


