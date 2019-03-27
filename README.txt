Trenton Rochelle
9561606

This liptracking assignment can be run by doing the following:

At the bottom of FINAL.py are three main functions which run the program. The main function takes the parameters as told by the worksheet.
The main functions will run one after another for each of the testimages. If you only want to see one, comment out the other two.
The outputs will be saved to a created folder in the current directory. The folder is named "output_" + filenamebase

ex) testimages/testimage2_02354.jpg will be saved to output_testimage2

This program runs a little slow because I couldn't get the alpha,beta, and lambda quite perfect so I have more iterations but it runs as intended.
I tried to make the lip program more robust by filtering out the pink in the image. This turned out to be a harder task than I initially
thought because one of the testcase girls has pink makeup on her cheeks. Specifically, liptracking2 has this so I created a hard threshold.
The lip tracking on her will be a little large because the black and white image of her creates bigger lips.
I did pink-filtering because the grayscale image of liptracking4 was difficult to see where her lips started and her face skin ended.

Overall the program works but I had some parameter tuning issues that results in the template getting stuck on certain things such as face edges or teeth and sometimes it won't recover.
I can adjust the parameters for a given set and get it to run great on that particular set but it seems that my thing is not robust enough for large variation.

The arguments that my program works best are:

main("testimages/liptracking3/","liptracking3_01295.jpg",1295, 1595, template_lip3)
***The snake works amazing on this dataset. I only tried 300 on her but the program could go longer.


main("testimages/liptracking2/","liptracking2_00068.jpg",1302, 1493, template_lip2)
***The snake gets stuck on her face at some point and then after 1493 is spirals out of control


main("testimages/liptracking4/","liptracking4_00068.jpg",68, 85, template_lip4)
***this set fairs horribly due to her teeth showing constantly. The contour tries locks onto it preceeds to spiral into a small dot.
If I change the alpha and beta for this set, I can get a much better long run..


type 'python FINAL.py' to run