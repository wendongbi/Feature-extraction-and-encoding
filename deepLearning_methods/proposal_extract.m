fileFolder=fullfile('./Animals_with_Attributes2/JPEGImages/');
class_name = dir(fullfile(fileFolder, '*'));
count = 0;
proposal_num = [];
for i = 3:size(class_name)
    imgdir = ['./Animals_with_Attributes2/JPEGImages/', class_name(i).name, '/*.jpg'];
    imglist = dir(fullfile(imgdir));
    % fprintf('%s=====%d\n',imglist(1).name, size(imglist)); 
    for j = 1:size(imglist)
        imgPath = ['./Animals_with_Attributes2/JPEGImages/', class_name(i).name, '/', imglist(j).name];
        img_name = strcat(class_name(i).name,'_', num2str(10000 + j));
        tmp = get_proposal(imgPath,img_name);
        proposal_num(end+1) = tmp;
    end
    fprintf('%s extract success.%d', imglist(size(imglist)).name, size(proposal_num));
    count = count + size(imglist);
end
fprintf('%d\n', count);
proposal_num = proposal_num';
save 'proposal_num.txt' proposal_num1 -ascii;

