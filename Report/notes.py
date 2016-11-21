Q = 1.44 * 34 = 49 (34 = as average number of reviews)

What you can do, for instance is take the rate of reviews (w weighted mean), divide it by two (in order to reduce the scoring to a scale of [0,5] and add this value to 5(1−e−q). So the formula becomes
score=5p/10+5(1−e−q/Q)
where p is the review rating and q is the quantity of ratings and you chose for Q an appropriate number that shows what importance you attach to the notion "quantity". An example: An item has 3 times a revision score of 6 and 2 times a revision score of 7. Then p=(3.6+2.7)/5=6.4 if we take Q=10 then 5(1−e−5/10≈3.88) so the total score is 3.2+3.9=7.1 rounded 7. On the other hand if somebody has 20 scorings of 6 then p=6 and 5(1−e−20/10)≈4.58 so the final score is 3+4.6 rounded giving 8. The choice of Q depends on what you call "few", "moderate", "many". As a rule of thumb consider a value M that you consider "moderate" and take Q=−M/ln(1/2)≈1.44M. So if you think 100 is a moderate value the take Q=144. Finally you can also replace the equal weight on quantity an quality by a skewed one so that the final formula becomes:
score=Pp+10(1−P)(1−e−q/Q))
where P∈[0,1] (in the original formula we had P=0.5).


the play:

Train the data with via SVM with features without any thing

Train the data with SVM with features + social features

social network features:

calculate the top x reviewers. go through all reviews and change buisnesses feature for the "best reviewers score" 



SVM - accuracy with social_network_feature : 0.595588235294
SVM - accuracy without : 0.585058823529
SVM - accuracy with only the social_network_feature : 0.569647058824
Decision Tree - accuracy with social_network_feature : 0.697058823529
Decision Tree - accuracy without : 0.648
Decision Tree - accuracy with only the social_network_feature : 0.565117647059
RF Tree - accuracy with social_network_feature : 0.727117647059
RF Tree - accuracy without : 0.681470588235
RF Tree - accuracy with only the social_network_feature : 0.566529411765
MLP - accuracy with social_network_feature : 0.765705882353
MLP - accuracy without : 0.715294117647
MLP - accuracy with only the social_network_feature : 0.570764705882
Ada - accuracy with social_network_feature : 0.744470588235
Ada - accuracy without : 0.709
Ada - accuracy with only the social_network_feature : 0.590294117647
