       �K"	  �W���Abrain.Event:2n�p�]     7��	��W���A"��
r
MNIST_Input/xPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
q
MNIST_Input/y_Placeholder*
dtype0*'
_output_shapes
:���������
*
shape:���������

t
Input_Reshape/x_image/shapeConst*%
valueB"����         *
dtype0*
_output_shapes
:
�
Input_Reshape/x_imageReshapeMNIST_Input/xInput_Reshape/x_image/shape*/
_output_shapes
:���������*
T0*
Tshape0
s
Input_Reshape/input_img/tagConst*
dtype0*
_output_shapes
: *(
valueB BInput_Reshape/input_img
�
Input_Reshape/input_imgImageSummaryInput_Reshape/input_img/tagInput_Reshape/x_image*

max_images*
T0*
	bad_colorB:�  �*
_output_shapes
: 
}
$Conv1/weights/truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
h
#Conv1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
%Conv1/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
.Conv1/weights/truncated_normal/TruncatedNormalTruncatedNormal$Conv1/weights/truncated_normal/shape*
T0*
dtype0*&
_output_shapes
: *
seed2 *

seed 
�
"Conv1/weights/truncated_normal/mulMul.Conv1/weights/truncated_normal/TruncatedNormal%Conv1/weights/truncated_normal/stddev*
T0*&
_output_shapes
: 
�
Conv1/weights/truncated_normalAdd"Conv1/weights/truncated_normal/mul#Conv1/weights/truncated_normal/mean*
T0*&
_output_shapes
: 
�
Conv1/weights/weight
VariableV2*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
�
Conv1/weights/weight/AssignAssignConv1/weights/weightConv1/weights/truncated_normal*
use_locking(*
T0*'
_class
loc:@Conv1/weights/weight*
validate_shape(*&
_output_shapes
: 
�
Conv1/weights/weight/readIdentityConv1/weights/weight*
T0*'
_class
loc:@Conv1/weights/weight*&
_output_shapes
: 
^
Conv1/weights/summaries/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
#Conv1/weights/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#Conv1/weights/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/rangeRange#Conv1/weights/summaries/range/startConv1/weights/summaries/Rank#Conv1/weights/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv1/weights/summaries/MeanMeanConv1/weights/weight/readConv1/weights/summaries/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
#Conv1/weights/summaries/mean_1/tagsConst*/
value&B$ BConv1/weights/summaries/mean_1*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/mean_1ScalarSummary#Conv1/weights/summaries/mean_1/tagsConv1/weights/summaries/Mean*
T0*
_output_shapes
: 
�
"Conv1/weights/summaries/stddev/subSubConv1/weights/weight/readConv1/weights/summaries/Mean*&
_output_shapes
: *
T0
�
%Conv1/weights/summaries/stddev/SquareSquare"Conv1/weights/summaries/stddev/sub*
T0*&
_output_shapes
: 
}
$Conv1/weights/summaries/stddev/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
�
#Conv1/weights/summaries/stddev/MeanMean%Conv1/weights/summaries/stddev/Square$Conv1/weights/summaries/stddev/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
#Conv1/weights/summaries/stddev/SqrtSqrt#Conv1/weights/summaries/stddev/Mean*
_output_shapes
: *
T0
�
%Conv1/weights/summaries/stddev_1/tagsConst*1
value(B& B Conv1/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
 Conv1/weights/summaries/stddev_1ScalarSummary%Conv1/weights/summaries/stddev_1/tags#Conv1/weights/summaries/stddev/Sqrt*
T0*
_output_shapes
: 
`
Conv1/weights/summaries/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
%Conv1/weights/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%Conv1/weights/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/range_1Range%Conv1/weights/summaries/range_1/startConv1/weights/summaries/Rank_1%Conv1/weights/summaries/range_1/delta*
_output_shapes
:*

Tidx0
�
Conv1/weights/summaries/MaxMaxConv1/weights/weight/readConv1/weights/summaries/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
"Conv1/weights/summaries/max_1/tagsConst*.
value%B# BConv1/weights/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/max_1ScalarSummary"Conv1/weights/summaries/max_1/tagsConv1/weights/summaries/Max*
T0*
_output_shapes
: 
`
Conv1/weights/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
g
%Conv1/weights/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%Conv1/weights/summaries/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/range_2Range%Conv1/weights/summaries/range_2/startConv1/weights/summaries/Rank_2%Conv1/weights/summaries/range_2/delta*
_output_shapes
:*

Tidx0
�
Conv1/weights/summaries/MinMinConv1/weights/weight/readConv1/weights/summaries/range_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
"Conv1/weights/summaries/min_1/tagsConst*.
value%B# BConv1/weights/summaries/min_1*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/min_1ScalarSummary"Conv1/weights/summaries/min_1/tagsConv1/weights/summaries/Min*
_output_shapes
: *
T0
�
%Conv1/weights/summaries/histogram/tagConst*2
value)B' B!Conv1/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
!Conv1/weights/summaries/histogramHistogramSummary%Conv1/weights/summaries/histogram/tagConv1/weights/weight/read*
T0*
_output_shapes
: 
_
Conv1/biases/ConstConst*
valueB *���=*
dtype0*
_output_shapes
: 
}
Conv1/biases/bias
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Conv1/biases/bias/AssignAssignConv1/biases/biasConv1/biases/Const*
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
Conv1/biases/bias/readIdentityConv1/biases/bias*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
]
Conv1/biases/summaries/RankConst*
dtype0*
_output_shapes
: *
value	B :
d
"Conv1/biases/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Conv1/biases/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/rangeRange"Conv1/biases/summaries/range/startConv1/biases/summaries/Rank"Conv1/biases/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv1/biases/summaries/MeanMeanConv1/biases/bias/readConv1/biases/summaries/range*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
"Conv1/biases/summaries/mean_1/tagsConst*.
value%B# BConv1/biases/summaries/mean_1*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/mean_1ScalarSummary"Conv1/biases/summaries/mean_1/tagsConv1/biases/summaries/Mean*
T0*
_output_shapes
: 
�
!Conv1/biases/summaries/stddev/subSubConv1/biases/bias/readConv1/biases/summaries/Mean*
T0*
_output_shapes
: 
v
$Conv1/biases/summaries/stddev/SquareSquare!Conv1/biases/summaries/stddev/sub*
T0*
_output_shapes
: 
m
#Conv1/biases/summaries/stddev/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"Conv1/biases/summaries/stddev/MeanMean$Conv1/biases/summaries/stddev/Square#Conv1/biases/summaries/stddev/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
o
"Conv1/biases/summaries/stddev/SqrtSqrt"Conv1/biases/summaries/stddev/Mean*
_output_shapes
: *
T0
�
$Conv1/biases/summaries/stddev_1/tagsConst*0
value'B% BConv1/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/stddev_1ScalarSummary$Conv1/biases/summaries/stddev_1/tags"Conv1/biases/summaries/stddev/Sqrt*
_output_shapes
: *
T0
_
Conv1/biases/summaries/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
f
$Conv1/biases/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/range_1Range$Conv1/biases/summaries/range_1/startConv1/biases/summaries/Rank_1$Conv1/biases/summaries/range_1/delta*
_output_shapes
:*

Tidx0
�
Conv1/biases/summaries/MaxMaxConv1/biases/bias/readConv1/biases/summaries/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
~
!Conv1/biases/summaries/max_1/tagsConst*-
value$B" BConv1/biases/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/max_1ScalarSummary!Conv1/biases/summaries/max_1/tagsConv1/biases/summaries/Max*
T0*
_output_shapes
: 
_
Conv1/biases/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/range_2Range$Conv1/biases/summaries/range_2/startConv1/biases/summaries/Rank_2$Conv1/biases/summaries/range_2/delta*
_output_shapes
:*

Tidx0
�
Conv1/biases/summaries/MinMinConv1/biases/bias/readConv1/biases/summaries/range_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
~
!Conv1/biases/summaries/min_1/tagsConst*-
value$B" BConv1/biases/summaries/min_1*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/min_1ScalarSummary!Conv1/biases/summaries/min_1/tagsConv1/biases/summaries/Min*
T0*
_output_shapes
: 
�
$Conv1/biases/summaries/histogram/tagConst*1
value(B& B Conv1/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
 Conv1/biases/summaries/histogramHistogramSummary$Conv1/biases/summaries/histogram/tagConv1/biases/bias/read*
T0*
_output_shapes
: 
�
Conv1/conv2dConv2DInput_Reshape/x_imageConv1/weights/weight/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:��������� 
p
	Conv1/addAddConv1/conv2dConv1/biases/bias/read*
T0*/
_output_shapes
:��������� 
e
Conv1/conv1_wx_b/tagConst*
dtype0*
_output_shapes
: *!
valueB BConv1/conv1_wx_b
f
Conv1/conv1_wx_bHistogramSummaryConv1/conv1_wx_b/tag	Conv1/add*
T0*
_output_shapes
: 
W

Conv1/reluRelu	Conv1/add*
T0*/
_output_shapes
:��������� 
_
Conv1/h_conv1/tagConst*
dtype0*
_output_shapes
: *
valueB BConv1/h_conv1
a
Conv1/h_conv1HistogramSummaryConv1/h_conv1/tag
Conv1/relu*
T0*
_output_shapes
: 
�

Conv1/poolMaxPool
Conv1/relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:��������� *
T0
}
$Conv2/weights/truncated_normal/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
h
#Conv2/weights/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
j
%Conv2/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
.Conv2/weights/truncated_normal/TruncatedNormalTruncatedNormal$Conv2/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
: @*
seed2 
�
"Conv2/weights/truncated_normal/mulMul.Conv2/weights/truncated_normal/TruncatedNormal%Conv2/weights/truncated_normal/stddev*&
_output_shapes
: @*
T0
�
Conv2/weights/truncated_normalAdd"Conv2/weights/truncated_normal/mul#Conv2/weights/truncated_normal/mean*
T0*&
_output_shapes
: @
�
Conv2/weights/weight
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
�
Conv2/weights/weight/AssignAssignConv2/weights/weightConv2/weights/truncated_normal*
T0*'
_class
loc:@Conv2/weights/weight*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
Conv2/weights/weight/readIdentityConv2/weights/weight*
T0*'
_class
loc:@Conv2/weights/weight*&
_output_shapes
: @
^
Conv2/weights/summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
#Conv2/weights/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#Conv2/weights/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/rangeRange#Conv2/weights/summaries/range/startConv2/weights/summaries/Rank#Conv2/weights/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv2/weights/summaries/MeanMeanConv2/weights/weight/readConv2/weights/summaries/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
#Conv2/weights/summaries/mean_1/tagsConst*/
value&B$ BConv2/weights/summaries/mean_1*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/mean_1ScalarSummary#Conv2/weights/summaries/mean_1/tagsConv2/weights/summaries/Mean*
_output_shapes
: *
T0
�
"Conv2/weights/summaries/stddev/subSubConv2/weights/weight/readConv2/weights/summaries/Mean*
T0*&
_output_shapes
: @
�
%Conv2/weights/summaries/stddev/SquareSquare"Conv2/weights/summaries/stddev/sub*&
_output_shapes
: @*
T0
}
$Conv2/weights/summaries/stddev/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
�
#Conv2/weights/summaries/stddev/MeanMean%Conv2/weights/summaries/stddev/Square$Conv2/weights/summaries/stddev/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
#Conv2/weights/summaries/stddev/SqrtSqrt#Conv2/weights/summaries/stddev/Mean*
_output_shapes
: *
T0
�
%Conv2/weights/summaries/stddev_1/tagsConst*
dtype0*
_output_shapes
: *1
value(B& B Conv2/weights/summaries/stddev_1
�
 Conv2/weights/summaries/stddev_1ScalarSummary%Conv2/weights/summaries/stddev_1/tags#Conv2/weights/summaries/stddev/Sqrt*
_output_shapes
: *
T0
`
Conv2/weights/summaries/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
%Conv2/weights/summaries/range_1/startConst*
dtype0*
_output_shapes
: *
value	B : 
g
%Conv2/weights/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/range_1Range%Conv2/weights/summaries/range_1/startConv2/weights/summaries/Rank_1%Conv2/weights/summaries/range_1/delta*
_output_shapes
:*

Tidx0
�
Conv2/weights/summaries/MaxMaxConv2/weights/weight/readConv2/weights/summaries/range_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
"Conv2/weights/summaries/max_1/tagsConst*.
value%B# BConv2/weights/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/max_1ScalarSummary"Conv2/weights/summaries/max_1/tagsConv2/weights/summaries/Max*
T0*
_output_shapes
: 
`
Conv2/weights/summaries/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
g
%Conv2/weights/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%Conv2/weights/summaries/range_2/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Conv2/weights/summaries/range_2Range%Conv2/weights/summaries/range_2/startConv2/weights/summaries/Rank_2%Conv2/weights/summaries/range_2/delta*

Tidx0*
_output_shapes
:
�
Conv2/weights/summaries/MinMinConv2/weights/weight/readConv2/weights/summaries/range_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"Conv2/weights/summaries/min_1/tagsConst*.
value%B# BConv2/weights/summaries/min_1*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/min_1ScalarSummary"Conv2/weights/summaries/min_1/tagsConv2/weights/summaries/Min*
T0*
_output_shapes
: 
�
%Conv2/weights/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *2
value)B' B!Conv2/weights/summaries/histogram
�
!Conv2/weights/summaries/histogramHistogramSummary%Conv2/weights/summaries/histogram/tagConv2/weights/weight/read*
T0*
_output_shapes
: 
_
Conv2/biases/ConstConst*
valueB@*���=*
dtype0*
_output_shapes
:@
}
Conv2/biases/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
�
Conv2/biases/bias/AssignAssignConv2/biases/biasConv2/biases/Const*
use_locking(*
T0*$
_class
loc:@Conv2/biases/bias*
validate_shape(*
_output_shapes
:@
�
Conv2/biases/bias/readIdentityConv2/biases/bias*
T0*$
_class
loc:@Conv2/biases/bias*
_output_shapes
:@
]
Conv2/biases/summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
d
"Conv2/biases/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Conv2/biases/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/rangeRange"Conv2/biases/summaries/range/startConv2/biases/summaries/Rank"Conv2/biases/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv2/biases/summaries/MeanMeanConv2/biases/bias/readConv2/biases/summaries/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
"Conv2/biases/summaries/mean_1/tagsConst*.
value%B# BConv2/biases/summaries/mean_1*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/mean_1ScalarSummary"Conv2/biases/summaries/mean_1/tagsConv2/biases/summaries/Mean*
T0*
_output_shapes
: 
�
!Conv2/biases/summaries/stddev/subSubConv2/biases/bias/readConv2/biases/summaries/Mean*
T0*
_output_shapes
:@
v
$Conv2/biases/summaries/stddev/SquareSquare!Conv2/biases/summaries/stddev/sub*
T0*
_output_shapes
:@
m
#Conv2/biases/summaries/stddev/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"Conv2/biases/summaries/stddev/MeanMean$Conv2/biases/summaries/stddev/Square#Conv2/biases/summaries/stddev/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
o
"Conv2/biases/summaries/stddev/SqrtSqrt"Conv2/biases/summaries/stddev/Mean*
T0*
_output_shapes
: 
�
$Conv2/biases/summaries/stddev_1/tagsConst*
dtype0*
_output_shapes
: *0
value'B% BConv2/biases/summaries/stddev_1
�
Conv2/biases/summaries/stddev_1ScalarSummary$Conv2/biases/summaries/stddev_1/tags"Conv2/biases/summaries/stddev/Sqrt*
T0*
_output_shapes
: 
_
Conv2/biases/summaries/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
f
$Conv2/biases/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/range_1Range$Conv2/biases/summaries/range_1/startConv2/biases/summaries/Rank_1$Conv2/biases/summaries/range_1/delta*

Tidx0*
_output_shapes
:
�
Conv2/biases/summaries/MaxMaxConv2/biases/bias/readConv2/biases/summaries/range_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
!Conv2/biases/summaries/max_1/tagsConst*-
value$B" BConv2/biases/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/max_1ScalarSummary!Conv2/biases/summaries/max_1/tagsConv2/biases/summaries/Max*
_output_shapes
: *
T0
_
Conv2/biases/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/range_2Range$Conv2/biases/summaries/range_2/startConv2/biases/summaries/Rank_2$Conv2/biases/summaries/range_2/delta*

Tidx0*
_output_shapes
:
�
Conv2/biases/summaries/MinMinConv2/biases/bias/readConv2/biases/summaries/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
!Conv2/biases/summaries/min_1/tagsConst*
dtype0*
_output_shapes
: *-
value$B" BConv2/biases/summaries/min_1
�
Conv2/biases/summaries/min_1ScalarSummary!Conv2/biases/summaries/min_1/tagsConv2/biases/summaries/Min*
T0*
_output_shapes
: 
�
$Conv2/biases/summaries/histogram/tagConst*1
value(B& B Conv2/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
 Conv2/biases/summaries/histogramHistogramSummary$Conv2/biases/summaries/histogram/tagConv2/biases/bias/read*
T0*
_output_shapes
: 
�
Conv2/conv2dConv2D
Conv1/poolConv2/weights/weight/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@
p
	Conv2/addAddConv2/conv2dConv2/biases/bias/read*
T0*/
_output_shapes
:���������@
e
Conv2/conv2_wx_b/tagConst*!
valueB BConv2/conv2_wx_b*
dtype0*
_output_shapes
: 
f
Conv2/conv2_wx_bHistogramSummaryConv2/conv2_wx_b/tag	Conv2/add*
T0*
_output_shapes
: 
W

Conv2/reluRelu	Conv2/add*
T0*/
_output_shapes
:���������@
_
Conv2/h_conv2/tagConst*
valueB BConv2/h_conv2*
dtype0*
_output_shapes
: 
a
Conv2/h_conv2HistogramSummaryConv2/h_conv2/tag
Conv2/relu*
_output_shapes
: *
T0
�

Conv2/poolMaxPool
Conv2/relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
j
FC/truncated_normal/shapeConst*
valueB"@     *
dtype0*
_output_shapes
:
]
FC/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
FC/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
#FC/truncated_normal/TruncatedNormalTruncatedNormalFC/truncated_normal/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2 *

seed 
�
FC/truncated_normal/mulMul#FC/truncated_normal/TruncatedNormalFC/truncated_normal/stddev*
T0* 
_output_shapes
:
��
x
FC/truncated_normalAddFC/truncated_normal/mulFC/truncated_normal/mean* 
_output_shapes
:
��*
T0
�
	FC/weight
VariableV2*
dtype0* 
_output_shapes
:
��*
	container *
shape:
��*
shared_name 
�
FC/weight/AssignAssign	FC/weightFC/truncated_normal*
use_locking(*
T0*
_class
loc:@FC/weight*
validate_shape(* 
_output_shapes
:
��
n
FC/weight/readIdentity	FC/weight*
T0*
_class
loc:@FC/weight* 
_output_shapes
:
��
W
FC/ConstConst*
valueB�*���=*
dtype0*
_output_shapes	
:�
u
FC/bias
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
FC/bias/AssignAssignFC/biasFC/Const*
use_locking(*
T0*
_class
loc:@FC/bias*
validate_shape(*
_output_shapes	
:�
c
FC/bias/readIdentityFC/bias*
T0*
_class
loc:@FC/bias*
_output_shapes	
:�
a
FC/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����@  
t

FC/ReshapeReshape
Conv2/poolFC/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
�
	FC/MatMulMatMul
FC/ReshapeFC/weight/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
Y
FC/addAdd	FC/MatMulFC/bias/read*
T0*(
_output_shapes
:����������
J
FC/reluReluFC/add*
T0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
T
dropout/ShapeShapeFC/relu*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*(
_output_shapes
:����������*
seed2 *

seed *
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
X
dropout/addAdd	keep_probdropout/random_uniform*
_output_shapes
:*
T0
F
dropout/FloorFloordropout/add*
_output_shapes
:*
T0
M
dropout/divRealDivFC/relu	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
o
Readout/truncated_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
b
Readout/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
Readout/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
(Readout/truncated_normal/TruncatedNormalTruncatedNormalReadout/truncated_normal/shape*
T0*
dtype0*
_output_shapes
:	�
*
seed2 *

seed 
�
Readout/truncated_normal/mulMul(Readout/truncated_normal/TruncatedNormalReadout/truncated_normal/stddev*
T0*
_output_shapes
:	�

�
Readout/truncated_normalAddReadout/truncated_normal/mulReadout/truncated_normal/mean*
T0*
_output_shapes
:	�

�
Readout/weight
VariableV2*
dtype0*
_output_shapes
:	�
*
	container *
shape:	�
*
shared_name 
�
Readout/weight/AssignAssignReadout/weightReadout/truncated_normal*
T0*!
_class
loc:@Readout/weight*
validate_shape(*
_output_shapes
:	�
*
use_locking(
|
Readout/weight/readIdentityReadout/weight*
T0*!
_class
loc:@Readout/weight*
_output_shapes
:	�

Z
Readout/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*���=
x
Readout/bias
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
Readout/bias/AssignAssignReadout/biasReadout/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Readout/bias
q
Readout/bias/readIdentityReadout/bias*
T0*
_class
loc:@Readout/bias*
_output_shapes
:

�
MatMulMatMuldropout/mulReadout/weight/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
W
addAddMatMulReadout/bias/read*
T0*'
_output_shapes
:���������

�
Gcross_entropy/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientMNIST_Input/y_*
T0*'
_output_shapes
:���������

y
7cross_entropy/softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
{
8cross_entropy/softmax_cross_entropy_with_logits_sg/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
{
9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
}
:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
z
8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
6cross_entropy/softmax_cross_entropy_with_logits_sg/SubSub9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_18cross_entropy/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
�
>cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/beginPack6cross_entropy/softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
�
=cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
8cross_entropy/softmax_cross_entropy_with_logits_sg/SliceSlice:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_1>cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/begin=cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
Index0*
T0
�
Bcross_entropy/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
�
>cross_entropy/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
9cross_entropy/softmax_cross_entropy_with_logits_sg/concatConcatV2Bcross_entropy/softmax_cross_entropy_with_logits_sg/concat/values_08cross_entropy/softmax_cross_entropy_with_logits_sg/Slice>cross_entropy/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/ReshapeReshapeadd9cross_entropy/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:������������������
{
9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_2ShapeGcross_entropy/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
|
:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1Sub9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_2:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
�
@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1*
N*
_output_shapes
:*
T0*

axis 
�
?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1Slice:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_2@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/begin?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
Dcross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
@cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
;cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2Dcross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/values_0:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1@cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_1ReshapeGcross_entropy/softmax_cross_entropy_with_logits_sg/labels_stop_gradient;cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
2cross_entropy/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits:cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
|
:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2Sub7cross_entropy/softmax_cross_entropy_with_logits_sg/Rank:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
�
@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2Slice8cross_entropy/softmax_cross_entropy_with_logits_sg/Shape@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/begin?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape2cross_entropy/softmax_cross_entropy_with_logits_sg:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
]
cross_entropy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
cross_entropy/MeanMean<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2cross_entropy/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
loss_optimizer/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
g
"loss_optimizer/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
loss_optimizer/gradients/FillFillloss_optimizer/gradients/Shape"loss_optimizer/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
>loss_optimizer/gradients/cross_entropy/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/ReshapeReshapeloss_optimizer/gradients/Fill>loss_optimizer/gradients/cross_entropy/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
6loss_optimizer/gradients/cross_entropy/Mean_grad/ShapeShape<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
�
5loss_optimizer/gradients/cross_entropy/Mean_grad/TileTile8loss_optimizer/gradients/cross_entropy/Mean_grad/Reshape6loss_optimizer/gradients/cross_entropy/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_1Shape<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
{
8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
�
6loss_optimizer/gradients/cross_entropy/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
5loss_optimizer/gradients/cross_entropy/Mean_grad/ProdProd8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_16loss_optimizer/gradients/cross_entropy/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
7loss_optimizer/gradients/cross_entropy/Mean_grad/Prod_1Prod8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_28loss_optimizer/gradients/cross_entropy/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
|
:loss_optimizer/gradients/cross_entropy/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/MaximumMaximum7loss_optimizer/gradients/cross_entropy/Mean_grad/Prod_1:loss_optimizer/gradients/cross_entropy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
9loss_optimizer/gradients/cross_entropy/Mean_grad/floordivFloorDiv5loss_optimizer/gradients/cross_entropy/Mean_grad/Prod8loss_optimizer/gradients/cross_entropy/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
5loss_optimizer/gradients/cross_entropy/Mean_grad/CastCast9loss_optimizer/gradients/cross_entropy/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/truedivRealDiv5loss_optimizer/gradients/cross_entropy/Mean_grad/Tile5loss_optimizer/gradients/cross_entropy/Mean_grad/Cast*#
_output_shapes
:���������*
T0
�
`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape2cross_entropy/softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
�
bloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape8loss_optimizer/gradients/cross_entropy/Mean_grad/truediv`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
#loss_optimizer/gradients/zeros_like	ZerosLike4cross_entropy/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
_loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsbloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshape_loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Tloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mulMul[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims4cross_entropy/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax:cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:������������������*
T0
�
Tloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/NegNeg[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:������������������
�
aloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsbloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshapealoss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Vloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul]loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Tloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/Neg*0
_output_shapes
:������������������*
T0
�
aloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpU^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mulW^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1
�
iloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityTloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mulb^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*0
_output_shapes
:������������������*
T0*g
_class]
[Yloc:@loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul
�
kloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityVloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1b^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*0
_output_shapes
:������������������*
T0*i
_class_
][loc:@loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1
�
^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
�
`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeiloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
m
'loss_optimizer/gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
s
)loss_optimizer/gradients/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
7loss_optimizer/gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgs'loss_optimizer/gradients/add_grad/Shape)loss_optimizer/gradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%loss_optimizer/gradients/add_grad/SumSum`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape7loss_optimizer/gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
)loss_optimizer/gradients/add_grad/ReshapeReshape%loss_optimizer/gradients/add_grad/Sum'loss_optimizer/gradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
'loss_optimizer/gradients/add_grad/Sum_1Sum`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape9loss_optimizer/gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
+loss_optimizer/gradients/add_grad/Reshape_1Reshape'loss_optimizer/gradients/add_grad/Sum_1)loss_optimizer/gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
2loss_optimizer/gradients/add_grad/tuple/group_depsNoOp*^loss_optimizer/gradients/add_grad/Reshape,^loss_optimizer/gradients/add_grad/Reshape_1
�
:loss_optimizer/gradients/add_grad/tuple/control_dependencyIdentity)loss_optimizer/gradients/add_grad/Reshape3^loss_optimizer/gradients/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@loss_optimizer/gradients/add_grad/Reshape*'
_output_shapes
:���������

�
<loss_optimizer/gradients/add_grad/tuple/control_dependency_1Identity+loss_optimizer/gradients/add_grad/Reshape_13^loss_optimizer/gradients/add_grad/tuple/group_deps*
_output_shapes
:
*
T0*>
_class4
20loc:@loss_optimizer/gradients/add_grad/Reshape_1
�
+loss_optimizer/gradients/MatMul_grad/MatMulMatMul:loss_optimizer/gradients/add_grad/tuple/control_dependencyReadout/weight/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
-loss_optimizer/gradients/MatMul_grad/MatMul_1MatMuldropout/mul:loss_optimizer/gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	�
*
transpose_a(*
transpose_b( 
�
5loss_optimizer/gradients/MatMul_grad/tuple/group_depsNoOp,^loss_optimizer/gradients/MatMul_grad/MatMul.^loss_optimizer/gradients/MatMul_grad/MatMul_1
�
=loss_optimizer/gradients/MatMul_grad/tuple/control_dependencyIdentity+loss_optimizer/gradients/MatMul_grad/MatMul6^loss_optimizer/gradients/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@loss_optimizer/gradients/MatMul_grad/MatMul*(
_output_shapes
:����������
�
?loss_optimizer/gradients/MatMul_grad/tuple/control_dependency_1Identity-loss_optimizer/gradients/MatMul_grad/MatMul_16^loss_optimizer/gradients/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@loss_optimizer/gradients/MatMul_grad/MatMul_1*
_output_shapes
:	�

�
/loss_optimizer/gradients/dropout/mul_grad/ShapeShapedropout/div*#
_output_shapes
:���������*
T0*
out_type0
�
1loss_optimizer/gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
_output_shapes
:���������
�
?loss_optimizer/gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs/loss_optimizer/gradients/dropout/mul_grad/Shape1loss_optimizer/gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
-loss_optimizer/gradients/dropout/mul_grad/MulMul=loss_optimizer/gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
�
-loss_optimizer/gradients/dropout/mul_grad/SumSum-loss_optimizer/gradients/dropout/mul_grad/Mul?loss_optimizer/gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1loss_optimizer/gradients/dropout/mul_grad/ReshapeReshape-loss_optimizer/gradients/dropout/mul_grad/Sum/loss_optimizer/gradients/dropout/mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
/loss_optimizer/gradients/dropout/mul_grad/Mul_1Muldropout/div=loss_optimizer/gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
/loss_optimizer/gradients/dropout/mul_grad/Sum_1Sum/loss_optimizer/gradients/dropout/mul_grad/Mul_1Aloss_optimizer/gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
3loss_optimizer/gradients/dropout/mul_grad/Reshape_1Reshape/loss_optimizer/gradients/dropout/mul_grad/Sum_11loss_optimizer/gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
:loss_optimizer/gradients/dropout/mul_grad/tuple/group_depsNoOp2^loss_optimizer/gradients/dropout/mul_grad/Reshape4^loss_optimizer/gradients/dropout/mul_grad/Reshape_1
�
Bloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependencyIdentity1loss_optimizer/gradients/dropout/mul_grad/Reshape;^loss_optimizer/gradients/dropout/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@loss_optimizer/gradients/dropout/mul_grad/Reshape*
_output_shapes
:
�
Dloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependency_1Identity3loss_optimizer/gradients/dropout/mul_grad/Reshape_1;^loss_optimizer/gradients/dropout/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@loss_optimizer/gradients/dropout/mul_grad/Reshape_1*
_output_shapes
:
v
/loss_optimizer/gradients/dropout/div_grad/ShapeShapeFC/relu*
T0*
out_type0*
_output_shapes
:
�
1loss_optimizer/gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:���������
�
?loss_optimizer/gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs/loss_optimizer/gradients/dropout/div_grad/Shape1loss_optimizer/gradients/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1loss_optimizer/gradients/dropout/div_grad/RealDivRealDivBloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
�
-loss_optimizer/gradients/dropout/div_grad/SumSum1loss_optimizer/gradients/dropout/div_grad/RealDiv?loss_optimizer/gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1loss_optimizer/gradients/dropout/div_grad/ReshapeReshape-loss_optimizer/gradients/dropout/div_grad/Sum/loss_optimizer/gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
p
-loss_optimizer/gradients/dropout/div_grad/NegNegFC/relu*
T0*(
_output_shapes
:����������
�
3loss_optimizer/gradients/dropout/div_grad/RealDiv_1RealDiv-loss_optimizer/gradients/dropout/div_grad/Neg	keep_prob*
_output_shapes
:*
T0
�
3loss_optimizer/gradients/dropout/div_grad/RealDiv_2RealDiv3loss_optimizer/gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
�
-loss_optimizer/gradients/dropout/div_grad/mulMulBloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependency3loss_optimizer/gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
�
/loss_optimizer/gradients/dropout/div_grad/Sum_1Sum-loss_optimizer/gradients/dropout/div_grad/mulAloss_optimizer/gradients/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3loss_optimizer/gradients/dropout/div_grad/Reshape_1Reshape/loss_optimizer/gradients/dropout/div_grad/Sum_11loss_optimizer/gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
:loss_optimizer/gradients/dropout/div_grad/tuple/group_depsNoOp2^loss_optimizer/gradients/dropout/div_grad/Reshape4^loss_optimizer/gradients/dropout/div_grad/Reshape_1
�
Bloss_optimizer/gradients/dropout/div_grad/tuple/control_dependencyIdentity1loss_optimizer/gradients/dropout/div_grad/Reshape;^loss_optimizer/gradients/dropout/div_grad/tuple/group_deps*
T0*D
_class:
86loc:@loss_optimizer/gradients/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
Dloss_optimizer/gradients/dropout/div_grad/tuple/control_dependency_1Identity3loss_optimizer/gradients/dropout/div_grad/Reshape_1;^loss_optimizer/gradients/dropout/div_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@loss_optimizer/gradients/dropout/div_grad/Reshape_1
�
.loss_optimizer/gradients/FC/relu_grad/ReluGradReluGradBloss_optimizer/gradients/dropout/div_grad/tuple/control_dependencyFC/relu*
T0*(
_output_shapes
:����������
s
*loss_optimizer/gradients/FC/add_grad/ShapeShape	FC/MatMul*
T0*
out_type0*
_output_shapes
:
w
,loss_optimizer/gradients/FC/add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
:loss_optimizer/gradients/FC/add_grad/BroadcastGradientArgsBroadcastGradientArgs*loss_optimizer/gradients/FC/add_grad/Shape,loss_optimizer/gradients/FC/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
(loss_optimizer/gradients/FC/add_grad/SumSum.loss_optimizer/gradients/FC/relu_grad/ReluGrad:loss_optimizer/gradients/FC/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
,loss_optimizer/gradients/FC/add_grad/ReshapeReshape(loss_optimizer/gradients/FC/add_grad/Sum*loss_optimizer/gradients/FC/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
*loss_optimizer/gradients/FC/add_grad/Sum_1Sum.loss_optimizer/gradients/FC/relu_grad/ReluGrad<loss_optimizer/gradients/FC/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
.loss_optimizer/gradients/FC/add_grad/Reshape_1Reshape*loss_optimizer/gradients/FC/add_grad/Sum_1,loss_optimizer/gradients/FC/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
5loss_optimizer/gradients/FC/add_grad/tuple/group_depsNoOp-^loss_optimizer/gradients/FC/add_grad/Reshape/^loss_optimizer/gradients/FC/add_grad/Reshape_1
�
=loss_optimizer/gradients/FC/add_grad/tuple/control_dependencyIdentity,loss_optimizer/gradients/FC/add_grad/Reshape6^loss_optimizer/gradients/FC/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@loss_optimizer/gradients/FC/add_grad/Reshape*(
_output_shapes
:����������
�
?loss_optimizer/gradients/FC/add_grad/tuple/control_dependency_1Identity.loss_optimizer/gradients/FC/add_grad/Reshape_16^loss_optimizer/gradients/FC/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss_optimizer/gradients/FC/add_grad/Reshape_1*
_output_shapes	
:�
�
.loss_optimizer/gradients/FC/MatMul_grad/MatMulMatMul=loss_optimizer/gradients/FC/add_grad/tuple/control_dependencyFC/weight/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
0loss_optimizer/gradients/FC/MatMul_grad/MatMul_1MatMul
FC/Reshape=loss_optimizer/gradients/FC/add_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
8loss_optimizer/gradients/FC/MatMul_grad/tuple/group_depsNoOp/^loss_optimizer/gradients/FC/MatMul_grad/MatMul1^loss_optimizer/gradients/FC/MatMul_grad/MatMul_1
�
@loss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependencyIdentity.loss_optimizer/gradients/FC/MatMul_grad/MatMul9^loss_optimizer/gradients/FC/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss_optimizer/gradients/FC/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Bloss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependency_1Identity0loss_optimizer/gradients/FC/MatMul_grad/MatMul_19^loss_optimizer/gradients/FC/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@loss_optimizer/gradients/FC/MatMul_grad/MatMul_1* 
_output_shapes
:
��
x
.loss_optimizer/gradients/FC/Reshape_grad/ShapeShape
Conv2/pool*
T0*
out_type0*
_output_shapes
:
�
0loss_optimizer/gradients/FC/Reshape_grad/ReshapeReshape@loss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependency.loss_optimizer/gradients/FC/Reshape_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
4loss_optimizer/gradients/Conv2/pool_grad/MaxPoolGradMaxPoolGrad
Conv2/relu
Conv2/pool0loss_optimizer/gradients/FC/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
�
1loss_optimizer/gradients/Conv2/relu_grad/ReluGradReluGrad4loss_optimizer/gradients/Conv2/pool_grad/MaxPoolGrad
Conv2/relu*
T0*/
_output_shapes
:���������@
y
-loss_optimizer/gradients/Conv2/add_grad/ShapeShapeConv2/conv2d*
T0*
out_type0*
_output_shapes
:
y
/loss_optimizer/gradients/Conv2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
�
=loss_optimizer/gradients/Conv2/add_grad/BroadcastGradientArgsBroadcastGradientArgs-loss_optimizer/gradients/Conv2/add_grad/Shape/loss_optimizer/gradients/Conv2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
+loss_optimizer/gradients/Conv2/add_grad/SumSum1loss_optimizer/gradients/Conv2/relu_grad/ReluGrad=loss_optimizer/gradients/Conv2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
/loss_optimizer/gradients/Conv2/add_grad/ReshapeReshape+loss_optimizer/gradients/Conv2/add_grad/Sum-loss_optimizer/gradients/Conv2/add_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
-loss_optimizer/gradients/Conv2/add_grad/Sum_1Sum1loss_optimizer/gradients/Conv2/relu_grad/ReluGrad?loss_optimizer/gradients/Conv2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
1loss_optimizer/gradients/Conv2/add_grad/Reshape_1Reshape-loss_optimizer/gradients/Conv2/add_grad/Sum_1/loss_optimizer/gradients/Conv2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
8loss_optimizer/gradients/Conv2/add_grad/tuple/group_depsNoOp0^loss_optimizer/gradients/Conv2/add_grad/Reshape2^loss_optimizer/gradients/Conv2/add_grad/Reshape_1
�
@loss_optimizer/gradients/Conv2/add_grad/tuple/control_dependencyIdentity/loss_optimizer/gradients/Conv2/add_grad/Reshape9^loss_optimizer/gradients/Conv2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@loss_optimizer/gradients/Conv2/add_grad/Reshape*/
_output_shapes
:���������@
�
Bloss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency_1Identity1loss_optimizer/gradients/Conv2/add_grad/Reshape_19^loss_optimizer/gradients/Conv2/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@loss_optimizer/gradients/Conv2/add_grad/Reshape_1*
_output_shapes
:@
�
1loss_optimizer/gradients/Conv2/conv2d_grad/ShapeNShapeN
Conv1/poolConv2/weights/weight/read*
N* 
_output_shapes
::*
T0*
out_type0
�
>loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput1loss_optimizer/gradients/Conv2/conv2d_grad/ShapeNConv2/weights/weight/read@loss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency*/
_output_shapes
:��������� *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
?loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilter
Conv1/pool3loss_optimizer/gradients/Conv2/conv2d_grad/ShapeN:1@loss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
�
;loss_optimizer/gradients/Conv2/conv2d_grad/tuple/group_depsNoOp@^loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilter?^loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInput
�
Closs_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependencyIdentity>loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInput<^loss_optimizer/gradients/Conv2/conv2d_grad/tuple/group_deps*/
_output_shapes
:��������� *
T0*Q
_classG
ECloc:@loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInput
�
Eloss_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependency_1Identity?loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilter<^loss_optimizer/gradients/Conv2/conv2d_grad/tuple/group_deps*
T0*R
_classH
FDloc:@loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
4loss_optimizer/gradients/Conv1/pool_grad/MaxPoolGradMaxPoolGrad
Conv1/relu
Conv1/poolCloss_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:��������� 
�
1loss_optimizer/gradients/Conv1/relu_grad/ReluGradReluGrad4loss_optimizer/gradients/Conv1/pool_grad/MaxPoolGrad
Conv1/relu*/
_output_shapes
:��������� *
T0
y
-loss_optimizer/gradients/Conv1/add_grad/ShapeShapeConv1/conv2d*
T0*
out_type0*
_output_shapes
:
y
/loss_optimizer/gradients/Conv1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
=loss_optimizer/gradients/Conv1/add_grad/BroadcastGradientArgsBroadcastGradientArgs-loss_optimizer/gradients/Conv1/add_grad/Shape/loss_optimizer/gradients/Conv1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+loss_optimizer/gradients/Conv1/add_grad/SumSum1loss_optimizer/gradients/Conv1/relu_grad/ReluGrad=loss_optimizer/gradients/Conv1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
/loss_optimizer/gradients/Conv1/add_grad/ReshapeReshape+loss_optimizer/gradients/Conv1/add_grad/Sum-loss_optimizer/gradients/Conv1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
-loss_optimizer/gradients/Conv1/add_grad/Sum_1Sum1loss_optimizer/gradients/Conv1/relu_grad/ReluGrad?loss_optimizer/gradients/Conv1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1loss_optimizer/gradients/Conv1/add_grad/Reshape_1Reshape-loss_optimizer/gradients/Conv1/add_grad/Sum_1/loss_optimizer/gradients/Conv1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
8loss_optimizer/gradients/Conv1/add_grad/tuple/group_depsNoOp0^loss_optimizer/gradients/Conv1/add_grad/Reshape2^loss_optimizer/gradients/Conv1/add_grad/Reshape_1
�
@loss_optimizer/gradients/Conv1/add_grad/tuple/control_dependencyIdentity/loss_optimizer/gradients/Conv1/add_grad/Reshape9^loss_optimizer/gradients/Conv1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@loss_optimizer/gradients/Conv1/add_grad/Reshape*/
_output_shapes
:��������� 
�
Bloss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency_1Identity1loss_optimizer/gradients/Conv1/add_grad/Reshape_19^loss_optimizer/gradients/Conv1/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@loss_optimizer/gradients/Conv1/add_grad/Reshape_1*
_output_shapes
: 
�
1loss_optimizer/gradients/Conv1/conv2d_grad/ShapeNShapeNInput_Reshape/x_imageConv1/weights/weight/read*
T0*
out_type0*
N* 
_output_shapes
::
�
>loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput1loss_optimizer/gradients/Conv1/conv2d_grad/ShapeNConv1/weights/weight/read@loss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
?loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterInput_Reshape/x_image3loss_optimizer/gradients/Conv1/conv2d_grad/ShapeN:1@loss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0
�
;loss_optimizer/gradients/Conv1/conv2d_grad/tuple/group_depsNoOp@^loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilter?^loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInput
�
Closs_optimizer/gradients/Conv1/conv2d_grad/tuple/control_dependencyIdentity>loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInput<^loss_optimizer/gradients/Conv1/conv2d_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*Q
_classG
ECloc:@loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInput
�
Eloss_optimizer/gradients/Conv1/conv2d_grad/tuple/control_dependency_1Identity?loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilter<^loss_optimizer/gradients/Conv1/conv2d_grad/tuple/group_deps*
T0*R
_classH
FDloc:@loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
(loss_optimizer/beta1_power/initial_valueConst*$
_class
loc:@Conv1/biases/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
loss_optimizer/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@Conv1/biases/bias
�
!loss_optimizer/beta1_power/AssignAssignloss_optimizer/beta1_power(loss_optimizer/beta1_power/initial_value*
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
loss_optimizer/beta1_power/readIdentityloss_optimizer/beta1_power*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
(loss_optimizer/beta2_power/initial_valueConst*$
_class
loc:@Conv1/biases/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
loss_optimizer/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@Conv1/biases/bias*
	container *
shape: 
�
!loss_optimizer/beta2_power/AssignAssignloss_optimizer/beta2_power(loss_optimizer/beta2_power/initial_value*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
loss_optimizer/beta2_power/readIdentityloss_optimizer/beta2_power*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
+Conv1/weights/weight/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
: *'
_class
loc:@Conv1/weights/weight*%
valueB *    
�
Conv1/weights/weight/Adam
VariableV2*
shared_name *'
_class
loc:@Conv1/weights/weight*
	container *
shape: *
dtype0*&
_output_shapes
: 
�
 Conv1/weights/weight/Adam/AssignAssignConv1/weights/weight/Adam+Conv1/weights/weight/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Conv1/weights/weight*
validate_shape(*&
_output_shapes
: 
�
Conv1/weights/weight/Adam/readIdentityConv1/weights/weight/Adam*
T0*'
_class
loc:@Conv1/weights/weight*&
_output_shapes
: 
�
-Conv1/weights/weight/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
: *'
_class
loc:@Conv1/weights/weight*%
valueB *    
�
Conv1/weights/weight/Adam_1
VariableV2*
shared_name *'
_class
loc:@Conv1/weights/weight*
	container *
shape: *
dtype0*&
_output_shapes
: 
�
"Conv1/weights/weight/Adam_1/AssignAssignConv1/weights/weight/Adam_1-Conv1/weights/weight/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Conv1/weights/weight
�
 Conv1/weights/weight/Adam_1/readIdentityConv1/weights/weight/Adam_1*&
_output_shapes
: *
T0*'
_class
loc:@Conv1/weights/weight
�
(Conv1/biases/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
: *$
_class
loc:@Conv1/biases/bias*
valueB *    
�
Conv1/biases/bias/Adam
VariableV2*$
_class
loc:@Conv1/biases/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Conv1/biases/bias/Adam/AssignAssignConv1/biases/bias/Adam(Conv1/biases/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias
�
Conv1/biases/bias/Adam/readIdentityConv1/biases/bias/Adam*
_output_shapes
: *
T0*$
_class
loc:@Conv1/biases/bias
�
*Conv1/biases/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@Conv1/biases/bias*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv1/biases/bias/Adam_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@Conv1/biases/bias*
	container 
�
Conv1/biases/bias/Adam_1/AssignAssignConv1/biases/bias/Adam_1*Conv1/biases/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
Conv1/biases/bias/Adam_1/readIdentityConv1/biases/bias/Adam_1*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
;Conv2/weights/weight/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Conv2/weights/weight*%
valueB"          @   *
dtype0*
_output_shapes
:
�
1Conv2/weights/weight/Adam/Initializer/zeros/ConstConst*'
_class
loc:@Conv2/weights/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
+Conv2/weights/weight/Adam/Initializer/zerosFill;Conv2/weights/weight/Adam/Initializer/zeros/shape_as_tensor1Conv2/weights/weight/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@Conv2/weights/weight*

index_type0*&
_output_shapes
: @
�
Conv2/weights/weight/Adam
VariableV2*
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *'
_class
loc:@Conv2/weights/weight*
	container 
�
 Conv2/weights/weight/Adam/AssignAssignConv2/weights/weight/Adam+Conv2/weights/weight/Adam/Initializer/zeros*
T0*'
_class
loc:@Conv2/weights/weight*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
Conv2/weights/weight/Adam/readIdentityConv2/weights/weight/Adam*
T0*'
_class
loc:@Conv2/weights/weight*&
_output_shapes
: @
�
=Conv2/weights/weight/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Conv2/weights/weight*%
valueB"          @   *
dtype0*
_output_shapes
:
�
3Conv2/weights/weight/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@Conv2/weights/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-Conv2/weights/weight/Adam_1/Initializer/zerosFill=Conv2/weights/weight/Adam_1/Initializer/zeros/shape_as_tensor3Conv2/weights/weight/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@Conv2/weights/weight*

index_type0*&
_output_shapes
: @
�
Conv2/weights/weight/Adam_1
VariableV2*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *'
_class
loc:@Conv2/weights/weight
�
"Conv2/weights/weight/Adam_1/AssignAssignConv2/weights/weight/Adam_1-Conv2/weights/weight/Adam_1/Initializer/zeros*
T0*'
_class
loc:@Conv2/weights/weight*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
 Conv2/weights/weight/Adam_1/readIdentityConv2/weights/weight/Adam_1*
T0*'
_class
loc:@Conv2/weights/weight*&
_output_shapes
: @
�
(Conv2/biases/bias/Adam/Initializer/zerosConst*$
_class
loc:@Conv2/biases/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2/biases/bias/Adam
VariableV2*
shared_name *$
_class
loc:@Conv2/biases/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
Conv2/biases/bias/Adam/AssignAssignConv2/biases/bias/Adam(Conv2/biases/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@Conv2/biases/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
Conv2/biases/bias/Adam/readIdentityConv2/biases/bias/Adam*
T0*$
_class
loc:@Conv2/biases/bias*
_output_shapes
:@
�
*Conv2/biases/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@Conv2/biases/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2/biases/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *$
_class
loc:@Conv2/biases/bias*
	container *
shape:@
�
Conv2/biases/bias/Adam_1/AssignAssignConv2/biases/bias/Adam_1*Conv2/biases/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Conv2/biases/bias*
validate_shape(*
_output_shapes
:@
�
Conv2/biases/bias/Adam_1/readIdentityConv2/biases/bias/Adam_1*
_output_shapes
:@*
T0*$
_class
loc:@Conv2/biases/bias
�
0FC/weight/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@FC/weight*
valueB"@     
�
&FC/weight/Adam/Initializer/zeros/ConstConst*
_class
loc:@FC/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 FC/weight/Adam/Initializer/zerosFill0FC/weight/Adam/Initializer/zeros/shape_as_tensor&FC/weight/Adam/Initializer/zeros/Const*
T0*
_class
loc:@FC/weight*

index_type0* 
_output_shapes
:
��
�
FC/weight/Adam
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *
_class
loc:@FC/weight
�
FC/weight/Adam/AssignAssignFC/weight/Adam FC/weight/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/weight*
validate_shape(* 
_output_shapes
:
��
x
FC/weight/Adam/readIdentityFC/weight/Adam*
T0*
_class
loc:@FC/weight* 
_output_shapes
:
��
�
2FC/weight/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@FC/weight*
valueB"@     *
dtype0*
_output_shapes
:
�
(FC/weight/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@FC/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"FC/weight/Adam_1/Initializer/zerosFill2FC/weight/Adam_1/Initializer/zeros/shape_as_tensor(FC/weight/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@FC/weight*

index_type0* 
_output_shapes
:
��
�
FC/weight/Adam_1
VariableV2*
_class
loc:@FC/weight*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
FC/weight/Adam_1/AssignAssignFC/weight/Adam_1"FC/weight/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/weight*
validate_shape(* 
_output_shapes
:
��
|
FC/weight/Adam_1/readIdentityFC/weight/Adam_1*
T0*
_class
loc:@FC/weight* 
_output_shapes
:
��
�
.FC/bias/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@FC/bias*
valueB:�*
dtype0*
_output_shapes
:
�
$FC/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@FC/bias*
valueB
 *    
�
FC/bias/Adam/Initializer/zerosFill.FC/bias/Adam/Initializer/zeros/shape_as_tensor$FC/bias/Adam/Initializer/zeros/Const*
T0*
_class
loc:@FC/bias*

index_type0*
_output_shapes	
:�
�
FC/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@FC/bias*
	container *
shape:�
�
FC/bias/Adam/AssignAssignFC/bias/AdamFC/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/bias*
validate_shape(*
_output_shapes	
:�
m
FC/bias/Adam/readIdentityFC/bias/Adam*
T0*
_class
loc:@FC/bias*
_output_shapes	
:�
�
0FC/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@FC/bias*
valueB:�
�
&FC/bias/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@FC/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 FC/bias/Adam_1/Initializer/zerosFill0FC/bias/Adam_1/Initializer/zeros/shape_as_tensor&FC/bias/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@FC/bias*

index_type0*
_output_shapes	
:�
�
FC/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@FC/bias*
	container *
shape:�
�
FC/bias/Adam_1/AssignAssignFC/bias/Adam_1 FC/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/bias*
validate_shape(*
_output_shapes	
:�
q
FC/bias/Adam_1/readIdentityFC/bias/Adam_1*
T0*
_class
loc:@FC/bias*
_output_shapes	
:�
�
5Readout/weight/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
_class
loc:@Readout/weight*
valueB"   
   
�
+Readout/weight/Adam/Initializer/zeros/ConstConst*!
_class
loc:@Readout/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%Readout/weight/Adam/Initializer/zerosFill5Readout/weight/Adam/Initializer/zeros/shape_as_tensor+Readout/weight/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@Readout/weight*

index_type0*
_output_shapes
:	�

�
Readout/weight/Adam
VariableV2*
dtype0*
_output_shapes
:	�
*
shared_name *!
_class
loc:@Readout/weight*
	container *
shape:	�

�
Readout/weight/Adam/AssignAssignReadout/weight/Adam%Readout/weight/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Readout/weight*
validate_shape(*
_output_shapes
:	�

�
Readout/weight/Adam/readIdentityReadout/weight/Adam*
_output_shapes
:	�
*
T0*!
_class
loc:@Readout/weight
�
7Readout/weight/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@Readout/weight*
valueB"   
   *
dtype0*
_output_shapes
:
�
-Readout/weight/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@Readout/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'Readout/weight/Adam_1/Initializer/zerosFill7Readout/weight/Adam_1/Initializer/zeros/shape_as_tensor-Readout/weight/Adam_1/Initializer/zeros/Const*
T0*!
_class
loc:@Readout/weight*

index_type0*
_output_shapes
:	�

�
Readout/weight/Adam_1
VariableV2*
dtype0*
_output_shapes
:	�
*
shared_name *!
_class
loc:@Readout/weight*
	container *
shape:	�

�
Readout/weight/Adam_1/AssignAssignReadout/weight/Adam_1'Readout/weight/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Readout/weight*
validate_shape(*
_output_shapes
:	�

�
Readout/weight/Adam_1/readIdentityReadout/weight/Adam_1*
T0*!
_class
loc:@Readout/weight*
_output_shapes
:	�

�
#Readout/bias/Adam/Initializer/zerosConst*
_class
loc:@Readout/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
Readout/bias/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@Readout/bias*
	container *
shape:

�
Readout/bias/Adam/AssignAssignReadout/bias/Adam#Readout/bias/Adam/Initializer/zeros*
T0*
_class
loc:@Readout/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
{
Readout/bias/Adam/readIdentityReadout/bias/Adam*
T0*
_class
loc:@Readout/bias*
_output_shapes
:

�
%Readout/bias/Adam_1/Initializer/zerosConst*
_class
loc:@Readout/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
Readout/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@Readout/bias*
	container *
shape:

�
Readout/bias/Adam_1/AssignAssignReadout/bias/Adam_1%Readout/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Readout/bias*
validate_shape(*
_output_shapes
:


Readout/bias/Adam_1/readIdentityReadout/bias/Adam_1*
T0*
_class
loc:@Readout/bias*
_output_shapes
:

f
!loss_optimizer/Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
^
loss_optimizer/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
^
loss_optimizer/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
`
loss_optimizer/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
9loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam	ApplyAdamConv1/weights/weightConv1/weights/weight/AdamConv1/weights/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonEloss_optimizer/gradients/Conv1/conv2d_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@Conv1/weights/weight*
use_nesterov( *&
_output_shapes
: 
�
6loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam	ApplyAdamConv1/biases/biasConv1/biases/bias/AdamConv1/biases/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonBloss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency_1*
T0*$
_class
loc:@Conv1/biases/bias*
use_nesterov( *
_output_shapes
: *
use_locking( 
�
9loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam	ApplyAdamConv2/weights/weightConv2/weights/weight/AdamConv2/weights/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonEloss_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@Conv2/weights/weight*
use_nesterov( *&
_output_shapes
: @
�
6loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam	ApplyAdamConv2/biases/biasConv2/biases/bias/AdamConv2/biases/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonBloss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Conv2/biases/bias*
use_nesterov( *
_output_shapes
:@
�
.loss_optimizer/Adam/update_FC/weight/ApplyAdam	ApplyAdam	FC/weightFC/weight/AdamFC/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonBloss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*
_class
loc:@FC/weight
�
,loss_optimizer/Adam/update_FC/bias/ApplyAdam	ApplyAdamFC/biasFC/bias/AdamFC/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilon?loss_optimizer/gradients/FC/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@FC/bias*
use_nesterov( *
_output_shapes	
:�
�
3loss_optimizer/Adam/update_Readout/weight/ApplyAdam	ApplyAdamReadout/weightReadout/weight/AdamReadout/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilon?loss_optimizer/gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Readout/weight*
use_nesterov( *
_output_shapes
:	�

�
1loss_optimizer/Adam/update_Readout/bias/ApplyAdam	ApplyAdamReadout/biasReadout/bias/AdamReadout/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilon<loss_optimizer/gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Readout/bias
�
loss_optimizer/Adam/mulMulloss_optimizer/beta1_power/readloss_optimizer/Adam/beta17^loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam7^loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam-^loss_optimizer/Adam/update_FC/bias/ApplyAdam/^loss_optimizer/Adam/update_FC/weight/ApplyAdam2^loss_optimizer/Adam/update_Readout/bias/ApplyAdam4^loss_optimizer/Adam/update_Readout/weight/ApplyAdam*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
loss_optimizer/Adam/AssignAssignloss_optimizer/beta1_powerloss_optimizer/Adam/mul*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�
loss_optimizer/Adam/mul_1Mulloss_optimizer/beta2_power/readloss_optimizer/Adam/beta27^loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam7^loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam-^loss_optimizer/Adam/update_FC/bias/ApplyAdam/^loss_optimizer/Adam/update_FC/weight/ApplyAdam2^loss_optimizer/Adam/update_Readout/bias/ApplyAdam4^loss_optimizer/Adam/update_Readout/weight/ApplyAdam*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
loss_optimizer/Adam/Assign_1Assignloss_optimizer/beta2_powerloss_optimizer/Adam/mul_1*
use_locking( *
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
loss_optimizer/AdamNoOp^loss_optimizer/Adam/Assign^loss_optimizer/Adam/Assign_17^loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam7^loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam-^loss_optimizer/Adam/update_FC/bias/ApplyAdam/^loss_optimizer/Adam/update_FC/weight/ApplyAdam2^loss_optimizer/Adam/update_Readout/bias/ApplyAdam4^loss_optimizer/Adam/update_Readout/weight/ApplyAdam
[
accuracy/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMaxArgMaxaddaccuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
]
accuracy/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMax_1ArgMaxMNIST_Input/y_accuracy/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
i
accuracy/EqualEqualaccuracy/ArgMaxaccuracy/ArgMax_1*
T0	*#
_output_shapes
:���������
r
accuracy/CastCastaccuracy/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
X
accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
accuracy/MeanMeanaccuracy/Castaccuracy/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
h
cross_entropy_scl/tagsConst*"
valueB Bcross_entropy_scl*
dtype0*
_output_shapes
: 
o
cross_entropy_sclScalarSummarycross_entropy_scl/tagscross_entropy/Mean*
T0*
_output_shapes
: 
h
training_accuracy/tagsConst*"
valueB Btraining_accuracy*
dtype0*
_output_shapes
: 
j
training_accuracyScalarSummarytraining_accuracy/tagsaccuracy/Mean*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummaryInput_Reshape/input_imgConv1/weights/summaries/mean_1 Conv1/weights/summaries/stddev_1Conv1/weights/summaries/max_1Conv1/weights/summaries/min_1!Conv1/weights/summaries/histogramConv1/biases/summaries/mean_1Conv1/biases/summaries/stddev_1Conv1/biases/summaries/max_1Conv1/biases/summaries/min_1 Conv1/biases/summaries/histogramConv1/conv1_wx_bConv1/h_conv1Conv2/weights/summaries/mean_1 Conv2/weights/summaries/stddev_1Conv2/weights/summaries/max_1Conv2/weights/summaries/min_1!Conv2/weights/summaries/histogramConv2/biases/summaries/mean_1Conv2/biases/summaries/stddev_1Conv2/biases/summaries/max_1Conv2/biases/summaries/min_1 Conv2/biases/summaries/histogramConv2/conv2_wx_bConv2/h_conv2cross_entropy_scltraining_accuracy*
N*
_output_shapes
: 
�
initNoOp^Conv1/biases/bias/Adam/Assign ^Conv1/biases/bias/Adam_1/Assign^Conv1/biases/bias/Assign!^Conv1/weights/weight/Adam/Assign#^Conv1/weights/weight/Adam_1/Assign^Conv1/weights/weight/Assign^Conv2/biases/bias/Adam/Assign ^Conv2/biases/bias/Adam_1/Assign^Conv2/biases/bias/Assign!^Conv2/weights/weight/Adam/Assign#^Conv2/weights/weight/Adam_1/Assign^Conv2/weights/weight/Assign^FC/bias/Adam/Assign^FC/bias/Adam_1/Assign^FC/bias/Assign^FC/weight/Adam/Assign^FC/weight/Adam_1/Assign^FC/weight/Assign^Readout/bias/Adam/Assign^Readout/bias/Adam_1/Assign^Readout/bias/Assign^Readout/weight/Adam/Assign^Readout/weight/Adam_1/Assign^Readout/weight/Assign"^loss_optimizer/beta1_power/Assign"^loss_optimizer/beta2_power/Assign"7fTX��     z<	���W���AJ��
�,�,
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
�
ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:�  �
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
�
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
2
StopGradient

input"T
output"T"	
Ttype
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.12.02v1.12.0-rc2-3-ga6d8ffae09��
r
MNIST_Input/xPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
q
MNIST_Input/y_Placeholder*
shape:���������
*
dtype0*'
_output_shapes
:���������

t
Input_Reshape/x_image/shapeConst*%
valueB"����         *
dtype0*
_output_shapes
:
�
Input_Reshape/x_imageReshapeMNIST_Input/xInput_Reshape/x_image/shape*/
_output_shapes
:���������*
T0*
Tshape0
s
Input_Reshape/input_img/tagConst*
dtype0*
_output_shapes
: *(
valueB BInput_Reshape/input_img
�
Input_Reshape/input_imgImageSummaryInput_Reshape/input_img/tagInput_Reshape/x_image*
_output_shapes
: *

max_images*
T0*
	bad_colorB:�  �
}
$Conv1/weights/truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
h
#Conv1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
%Conv1/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
.Conv1/weights/truncated_normal/TruncatedNormalTruncatedNormal$Conv1/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
: *
seed2 
�
"Conv1/weights/truncated_normal/mulMul.Conv1/weights/truncated_normal/TruncatedNormal%Conv1/weights/truncated_normal/stddev*
T0*&
_output_shapes
: 
�
Conv1/weights/truncated_normalAdd"Conv1/weights/truncated_normal/mul#Conv1/weights/truncated_normal/mean*
T0*&
_output_shapes
: 
�
Conv1/weights/weight
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
�
Conv1/weights/weight/AssignAssignConv1/weights/weightConv1/weights/truncated_normal*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Conv1/weights/weight
�
Conv1/weights/weight/readIdentityConv1/weights/weight*
T0*'
_class
loc:@Conv1/weights/weight*&
_output_shapes
: 
^
Conv1/weights/summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
#Conv1/weights/summaries/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
e
#Conv1/weights/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/rangeRange#Conv1/weights/summaries/range/startConv1/weights/summaries/Rank#Conv1/weights/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv1/weights/summaries/MeanMeanConv1/weights/weight/readConv1/weights/summaries/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
#Conv1/weights/summaries/mean_1/tagsConst*
dtype0*
_output_shapes
: */
value&B$ BConv1/weights/summaries/mean_1
�
Conv1/weights/summaries/mean_1ScalarSummary#Conv1/weights/summaries/mean_1/tagsConv1/weights/summaries/Mean*
T0*
_output_shapes
: 
�
"Conv1/weights/summaries/stddev/subSubConv1/weights/weight/readConv1/weights/summaries/Mean*
T0*&
_output_shapes
: 
�
%Conv1/weights/summaries/stddev/SquareSquare"Conv1/weights/summaries/stddev/sub*
T0*&
_output_shapes
: 
}
$Conv1/weights/summaries/stddev/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
�
#Conv1/weights/summaries/stddev/MeanMean%Conv1/weights/summaries/stddev/Square$Conv1/weights/summaries/stddev/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
#Conv1/weights/summaries/stddev/SqrtSqrt#Conv1/weights/summaries/stddev/Mean*
T0*
_output_shapes
: 
�
%Conv1/weights/summaries/stddev_1/tagsConst*1
value(B& B Conv1/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
 Conv1/weights/summaries/stddev_1ScalarSummary%Conv1/weights/summaries/stddev_1/tags#Conv1/weights/summaries/stddev/Sqrt*
T0*
_output_shapes
: 
`
Conv1/weights/summaries/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
g
%Conv1/weights/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%Conv1/weights/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/range_1Range%Conv1/weights/summaries/range_1/startConv1/weights/summaries/Rank_1%Conv1/weights/summaries/range_1/delta*
_output_shapes
:*

Tidx0
�
Conv1/weights/summaries/MaxMaxConv1/weights/weight/readConv1/weights/summaries/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
"Conv1/weights/summaries/max_1/tagsConst*
dtype0*
_output_shapes
: *.
value%B# BConv1/weights/summaries/max_1
�
Conv1/weights/summaries/max_1ScalarSummary"Conv1/weights/summaries/max_1/tagsConv1/weights/summaries/Max*
T0*
_output_shapes
: 
`
Conv1/weights/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
g
%Conv1/weights/summaries/range_2/startConst*
dtype0*
_output_shapes
: *
value	B : 
g
%Conv1/weights/summaries/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/weights/summaries/range_2Range%Conv1/weights/summaries/range_2/startConv1/weights/summaries/Rank_2%Conv1/weights/summaries/range_2/delta*

Tidx0*
_output_shapes
:
�
Conv1/weights/summaries/MinMinConv1/weights/weight/readConv1/weights/summaries/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
"Conv1/weights/summaries/min_1/tagsConst*
dtype0*
_output_shapes
: *.
value%B# BConv1/weights/summaries/min_1
�
Conv1/weights/summaries/min_1ScalarSummary"Conv1/weights/summaries/min_1/tagsConv1/weights/summaries/Min*
T0*
_output_shapes
: 
�
%Conv1/weights/summaries/histogram/tagConst*2
value)B' B!Conv1/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
!Conv1/weights/summaries/histogramHistogramSummary%Conv1/weights/summaries/histogram/tagConv1/weights/weight/read*
T0*
_output_shapes
: 
_
Conv1/biases/ConstConst*
dtype0*
_output_shapes
: *
valueB *���=
}
Conv1/biases/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Conv1/biases/bias/AssignAssignConv1/biases/biasConv1/biases/Const*
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
Conv1/biases/bias/readIdentityConv1/biases/bias*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
]
Conv1/biases/summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
d
"Conv1/biases/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Conv1/biases/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/rangeRange"Conv1/biases/summaries/range/startConv1/biases/summaries/Rank"Conv1/biases/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv1/biases/summaries/MeanMeanConv1/biases/bias/readConv1/biases/summaries/range*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
"Conv1/biases/summaries/mean_1/tagsConst*
dtype0*
_output_shapes
: *.
value%B# BConv1/biases/summaries/mean_1
�
Conv1/biases/summaries/mean_1ScalarSummary"Conv1/biases/summaries/mean_1/tagsConv1/biases/summaries/Mean*
T0*
_output_shapes
: 
�
!Conv1/biases/summaries/stddev/subSubConv1/biases/bias/readConv1/biases/summaries/Mean*
_output_shapes
: *
T0
v
$Conv1/biases/summaries/stddev/SquareSquare!Conv1/biases/summaries/stddev/sub*
T0*
_output_shapes
: 
m
#Conv1/biases/summaries/stddev/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"Conv1/biases/summaries/stddev/MeanMean$Conv1/biases/summaries/stddev/Square#Conv1/biases/summaries/stddev/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
o
"Conv1/biases/summaries/stddev/SqrtSqrt"Conv1/biases/summaries/stddev/Mean*
T0*
_output_shapes
: 
�
$Conv1/biases/summaries/stddev_1/tagsConst*
dtype0*
_output_shapes
: *0
value'B% BConv1/biases/summaries/stddev_1
�
Conv1/biases/summaries/stddev_1ScalarSummary$Conv1/biases/summaries/stddev_1/tags"Conv1/biases/summaries/stddev/Sqrt*
T0*
_output_shapes
: 
_
Conv1/biases/summaries/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/range_1Range$Conv1/biases/summaries/range_1/startConv1/biases/summaries/Rank_1$Conv1/biases/summaries/range_1/delta*

Tidx0*
_output_shapes
:
�
Conv1/biases/summaries/MaxMaxConv1/biases/bias/readConv1/biases/summaries/range_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
!Conv1/biases/summaries/max_1/tagsConst*-
value$B" BConv1/biases/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/max_1ScalarSummary!Conv1/biases/summaries/max_1/tagsConv1/biases/summaries/Max*
T0*
_output_shapes
: 
_
Conv1/biases/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv1/biases/summaries/range_2/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Conv1/biases/summaries/range_2Range$Conv1/biases/summaries/range_2/startConv1/biases/summaries/Rank_2$Conv1/biases/summaries/range_2/delta*
_output_shapes
:*

Tidx0
�
Conv1/biases/summaries/MinMinConv1/biases/bias/readConv1/biases/summaries/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
!Conv1/biases/summaries/min_1/tagsConst*-
value$B" BConv1/biases/summaries/min_1*
dtype0*
_output_shapes
: 
�
Conv1/biases/summaries/min_1ScalarSummary!Conv1/biases/summaries/min_1/tagsConv1/biases/summaries/Min*
T0*
_output_shapes
: 
�
$Conv1/biases/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *1
value(B& B Conv1/biases/summaries/histogram
�
 Conv1/biases/summaries/histogramHistogramSummary$Conv1/biases/summaries/histogram/tagConv1/biases/bias/read*
T0*
_output_shapes
: 
�
Conv1/conv2dConv2DInput_Reshape/x_imageConv1/weights/weight/read*
paddingSAME*/
_output_shapes
:��������� *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
p
	Conv1/addAddConv1/conv2dConv1/biases/bias/read*
T0*/
_output_shapes
:��������� 
e
Conv1/conv1_wx_b/tagConst*
dtype0*
_output_shapes
: *!
valueB BConv1/conv1_wx_b
f
Conv1/conv1_wx_bHistogramSummaryConv1/conv1_wx_b/tag	Conv1/add*
T0*
_output_shapes
: 
W

Conv1/reluRelu	Conv1/add*/
_output_shapes
:��������� *
T0
_
Conv1/h_conv1/tagConst*
valueB BConv1/h_conv1*
dtype0*
_output_shapes
: 
a
Conv1/h_conv1HistogramSummaryConv1/h_conv1/tag
Conv1/relu*
T0*
_output_shapes
: 
�

Conv1/poolMaxPool
Conv1/relu*/
_output_shapes
:��������� *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
}
$Conv2/weights/truncated_normal/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
h
#Conv2/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
%Conv2/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
.Conv2/weights/truncated_normal/TruncatedNormalTruncatedNormal$Conv2/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
: @*
seed2 
�
"Conv2/weights/truncated_normal/mulMul.Conv2/weights/truncated_normal/TruncatedNormal%Conv2/weights/truncated_normal/stddev*
T0*&
_output_shapes
: @
�
Conv2/weights/truncated_normalAdd"Conv2/weights/truncated_normal/mul#Conv2/weights/truncated_normal/mean*&
_output_shapes
: @*
T0
�
Conv2/weights/weight
VariableV2*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
�
Conv2/weights/weight/AssignAssignConv2/weights/weightConv2/weights/truncated_normal*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*'
_class
loc:@Conv2/weights/weight
�
Conv2/weights/weight/readIdentityConv2/weights/weight*
T0*'
_class
loc:@Conv2/weights/weight*&
_output_shapes
: @
^
Conv2/weights/summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
#Conv2/weights/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#Conv2/weights/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/rangeRange#Conv2/weights/summaries/range/startConv2/weights/summaries/Rank#Conv2/weights/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv2/weights/summaries/MeanMeanConv2/weights/weight/readConv2/weights/summaries/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
#Conv2/weights/summaries/mean_1/tagsConst*
dtype0*
_output_shapes
: */
value&B$ BConv2/weights/summaries/mean_1
�
Conv2/weights/summaries/mean_1ScalarSummary#Conv2/weights/summaries/mean_1/tagsConv2/weights/summaries/Mean*
T0*
_output_shapes
: 
�
"Conv2/weights/summaries/stddev/subSubConv2/weights/weight/readConv2/weights/summaries/Mean*
T0*&
_output_shapes
: @
�
%Conv2/weights/summaries/stddev/SquareSquare"Conv2/weights/summaries/stddev/sub*&
_output_shapes
: @*
T0
}
$Conv2/weights/summaries/stddev/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
�
#Conv2/weights/summaries/stddev/MeanMean%Conv2/weights/summaries/stddev/Square$Conv2/weights/summaries/stddev/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
#Conv2/weights/summaries/stddev/SqrtSqrt#Conv2/weights/summaries/stddev/Mean*
T0*
_output_shapes
: 
�
%Conv2/weights/summaries/stddev_1/tagsConst*
dtype0*
_output_shapes
: *1
value(B& B Conv2/weights/summaries/stddev_1
�
 Conv2/weights/summaries/stddev_1ScalarSummary%Conv2/weights/summaries/stddev_1/tags#Conv2/weights/summaries/stddev/Sqrt*
T0*
_output_shapes
: 
`
Conv2/weights/summaries/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
%Conv2/weights/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%Conv2/weights/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/range_1Range%Conv2/weights/summaries/range_1/startConv2/weights/summaries/Rank_1%Conv2/weights/summaries/range_1/delta*

Tidx0*
_output_shapes
:
�
Conv2/weights/summaries/MaxMaxConv2/weights/weight/readConv2/weights/summaries/range_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"Conv2/weights/summaries/max_1/tagsConst*.
value%B# BConv2/weights/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/max_1ScalarSummary"Conv2/weights/summaries/max_1/tagsConv2/weights/summaries/Max*
_output_shapes
: *
T0
`
Conv2/weights/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
g
%Conv2/weights/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%Conv2/weights/summaries/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/range_2Range%Conv2/weights/summaries/range_2/startConv2/weights/summaries/Rank_2%Conv2/weights/summaries/range_2/delta*
_output_shapes
:*

Tidx0
�
Conv2/weights/summaries/MinMinConv2/weights/weight/readConv2/weights/summaries/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
"Conv2/weights/summaries/min_1/tagsConst*.
value%B# BConv2/weights/summaries/min_1*
dtype0*
_output_shapes
: 
�
Conv2/weights/summaries/min_1ScalarSummary"Conv2/weights/summaries/min_1/tagsConv2/weights/summaries/Min*
T0*
_output_shapes
: 
�
%Conv2/weights/summaries/histogram/tagConst*2
value)B' B!Conv2/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
!Conv2/weights/summaries/histogramHistogramSummary%Conv2/weights/summaries/histogram/tagConv2/weights/weight/read*
T0*
_output_shapes
: 
_
Conv2/biases/ConstConst*
valueB@*���=*
dtype0*
_output_shapes
:@
}
Conv2/biases/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
�
Conv2/biases/bias/AssignAssignConv2/biases/biasConv2/biases/Const*
use_locking(*
T0*$
_class
loc:@Conv2/biases/bias*
validate_shape(*
_output_shapes
:@
�
Conv2/biases/bias/readIdentityConv2/biases/bias*
T0*$
_class
loc:@Conv2/biases/bias*
_output_shapes
:@
]
Conv2/biases/summaries/RankConst*
value	B :*
dtype0*
_output_shapes
: 
d
"Conv2/biases/summaries/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Conv2/biases/summaries/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/rangeRange"Conv2/biases/summaries/range/startConv2/biases/summaries/Rank"Conv2/biases/summaries/range/delta*
_output_shapes
:*

Tidx0
�
Conv2/biases/summaries/MeanMeanConv2/biases/bias/readConv2/biases/summaries/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
"Conv2/biases/summaries/mean_1/tagsConst*.
value%B# BConv2/biases/summaries/mean_1*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/mean_1ScalarSummary"Conv2/biases/summaries/mean_1/tagsConv2/biases/summaries/Mean*
T0*
_output_shapes
: 
�
!Conv2/biases/summaries/stddev/subSubConv2/biases/bias/readConv2/biases/summaries/Mean*
T0*
_output_shapes
:@
v
$Conv2/biases/summaries/stddev/SquareSquare!Conv2/biases/summaries/stddev/sub*
T0*
_output_shapes
:@
m
#Conv2/biases/summaries/stddev/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"Conv2/biases/summaries/stddev/MeanMean$Conv2/biases/summaries/stddev/Square#Conv2/biases/summaries/stddev/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
o
"Conv2/biases/summaries/stddev/SqrtSqrt"Conv2/biases/summaries/stddev/Mean*
T0*
_output_shapes
: 
�
$Conv2/biases/summaries/stddev_1/tagsConst*
dtype0*
_output_shapes
: *0
value'B% BConv2/biases/summaries/stddev_1
�
Conv2/biases/summaries/stddev_1ScalarSummary$Conv2/biases/summaries/stddev_1/tags"Conv2/biases/summaries/stddev/Sqrt*
_output_shapes
: *
T0
_
Conv2/biases/summaries/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/range_1Range$Conv2/biases/summaries/range_1/startConv2/biases/summaries/Rank_1$Conv2/biases/summaries/range_1/delta*
_output_shapes
:*

Tidx0
�
Conv2/biases/summaries/MaxMaxConv2/biases/bias/readConv2/biases/summaries/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
~
!Conv2/biases/summaries/max_1/tagsConst*-
value$B" BConv2/biases/summaries/max_1*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/max_1ScalarSummary!Conv2/biases/summaries/max_1/tagsConv2/biases/summaries/Max*
T0*
_output_shapes
: 
_
Conv2/biases/summaries/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$Conv2/biases/summaries/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/range_2Range$Conv2/biases/summaries/range_2/startConv2/biases/summaries/Rank_2$Conv2/biases/summaries/range_2/delta*

Tidx0*
_output_shapes
:
�
Conv2/biases/summaries/MinMinConv2/biases/bias/readConv2/biases/summaries/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
!Conv2/biases/summaries/min_1/tagsConst*-
value$B" BConv2/biases/summaries/min_1*
dtype0*
_output_shapes
: 
�
Conv2/biases/summaries/min_1ScalarSummary!Conv2/biases/summaries/min_1/tagsConv2/biases/summaries/Min*
T0*
_output_shapes
: 
�
$Conv2/biases/summaries/histogram/tagConst*1
value(B& B Conv2/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
 Conv2/biases/summaries/histogramHistogramSummary$Conv2/biases/summaries/histogram/tagConv2/biases/bias/read*
T0*
_output_shapes
: 
�
Conv2/conv2dConv2D
Conv1/poolConv2/weights/weight/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@
p
	Conv2/addAddConv2/conv2dConv2/biases/bias/read*/
_output_shapes
:���������@*
T0
e
Conv2/conv2_wx_b/tagConst*!
valueB BConv2/conv2_wx_b*
dtype0*
_output_shapes
: 
f
Conv2/conv2_wx_bHistogramSummaryConv2/conv2_wx_b/tag	Conv2/add*
T0*
_output_shapes
: 
W

Conv2/reluRelu	Conv2/add*
T0*/
_output_shapes
:���������@
_
Conv2/h_conv2/tagConst*
valueB BConv2/h_conv2*
dtype0*
_output_shapes
: 
a
Conv2/h_conv2HistogramSummaryConv2/h_conv2/tag
Conv2/relu*
_output_shapes
: *
T0
�

Conv2/poolMaxPool
Conv2/relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
j
FC/truncated_normal/shapeConst*
valueB"@     *
dtype0*
_output_shapes
:
]
FC/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
FC/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
#FC/truncated_normal/TruncatedNormalTruncatedNormalFC/truncated_normal/shape*
dtype0* 
_output_shapes
:
��*
seed2 *

seed *
T0
�
FC/truncated_normal/mulMul#FC/truncated_normal/TruncatedNormalFC/truncated_normal/stddev*
T0* 
_output_shapes
:
��
x
FC/truncated_normalAddFC/truncated_normal/mulFC/truncated_normal/mean*
T0* 
_output_shapes
:
��
�
	FC/weight
VariableV2*
shape:
��*
shared_name *
dtype0* 
_output_shapes
:
��*
	container 
�
FC/weight/AssignAssign	FC/weightFC/truncated_normal*
T0*
_class
loc:@FC/weight*
validate_shape(* 
_output_shapes
:
��*
use_locking(
n
FC/weight/readIdentity	FC/weight*
T0*
_class
loc:@FC/weight* 
_output_shapes
:
��
W
FC/ConstConst*
valueB�*���=*
dtype0*
_output_shapes	
:�
u
FC/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
FC/bias/AssignAssignFC/biasFC/Const*
T0*
_class
loc:@FC/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
c
FC/bias/readIdentityFC/bias*
T0*
_class
loc:@FC/bias*
_output_shapes	
:�
a
FC/Reshape/shapeConst*
valueB"����@  *
dtype0*
_output_shapes
:
t

FC/ReshapeReshape
Conv2/poolFC/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
	FC/MatMulMatMul
FC/ReshapeFC/weight/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
Y
FC/addAdd	FC/MatMulFC/bias/read*
T0*(
_output_shapes
:����������
J
FC/reluReluFC/add*
T0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
T
dropout/ShapeShapeFC/relu*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*(
_output_shapes
:����������*
seed2 *

seed *
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
T0
X
dropout/addAdd	keep_probdropout/random_uniform*
T0*
_output_shapes
:
F
dropout/FloorFloordropout/add*
_output_shapes
:*
T0
M
dropout/divRealDivFC/relu	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
o
Readout/truncated_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
b
Readout/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
Readout/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
(Readout/truncated_normal/TruncatedNormalTruncatedNormalReadout/truncated_normal/shape*

seed *
T0*
dtype0*
_output_shapes
:	�
*
seed2 
�
Readout/truncated_normal/mulMul(Readout/truncated_normal/TruncatedNormalReadout/truncated_normal/stddev*
T0*
_output_shapes
:	�

�
Readout/truncated_normalAddReadout/truncated_normal/mulReadout/truncated_normal/mean*
T0*
_output_shapes
:	�

�
Readout/weight
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�
*
	container *
shape:	�

�
Readout/weight/AssignAssignReadout/weightReadout/truncated_normal*
use_locking(*
T0*!
_class
loc:@Readout/weight*
validate_shape(*
_output_shapes
:	�

|
Readout/weight/readIdentityReadout/weight*
T0*!
_class
loc:@Readout/weight*
_output_shapes
:	�

Z
Readout/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*���=
x
Readout/bias
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
Readout/bias/AssignAssignReadout/biasReadout/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Readout/bias
q
Readout/bias/readIdentityReadout/bias*
T0*
_class
loc:@Readout/bias*
_output_shapes
:

�
MatMulMatMuldropout/mulReadout/weight/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
W
addAddMatMulReadout/bias/read*
T0*'
_output_shapes
:���������

�
Gcross_entropy/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientMNIST_Input/y_*'
_output_shapes
:���������
*
T0
y
7cross_entropy/softmax_cross_entropy_with_logits_sg/RankConst*
dtype0*
_output_shapes
: *
value	B :
{
8cross_entropy/softmax_cross_entropy_with_logits_sg/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
{
9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
}
:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_1Shapeadd*
_output_shapes
:*
T0*
out_type0
z
8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
6cross_entropy/softmax_cross_entropy_with_logits_sg/SubSub9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_18cross_entropy/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
�
>cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/beginPack6cross_entropy/softmax_cross_entropy_with_logits_sg/Sub*
N*
_output_shapes
:*
T0*

axis 
�
=cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
8cross_entropy/softmax_cross_entropy_with_logits_sg/SliceSlice:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_1>cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/begin=cross_entropy/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:
�
Bcross_entropy/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
�
>cross_entropy/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
9cross_entropy/softmax_cross_entropy_with_logits_sg/concatConcatV2Bcross_entropy/softmax_cross_entropy_with_logits_sg/concat/values_08cross_entropy/softmax_cross_entropy_with_logits_sg/Slice>cross_entropy/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/ReshapeReshapeadd9cross_entropy/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:������������������
{
9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_2ShapeGcross_entropy/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
|
:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1Sub9cross_entropy/softmax_cross_entropy_with_logits_sg/Rank_2:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
�
@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_1*
N*
_output_shapes
:*
T0*

axis 
�
?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1Slice:cross_entropy/softmax_cross_entropy_with_logits_sg/Shape_2@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/begin?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
Dcross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
@cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
;cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2Dcross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/values_0:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_1@cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_1ReshapeGcross_entropy/softmax_cross_entropy_with_logits_sg/labels_stop_gradient;cross_entropy/softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
2cross_entropy/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits:cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
|
:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2Sub7cross_entropy/softmax_cross_entropy_with_logits_sg/Rank:cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
�
@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack8cross_entropy/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2Slice8cross_entropy/softmax_cross_entropy_with_logits_sg/Shape@cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/begin?cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape2cross_entropy/softmax_cross_entropy_with_logits_sg:cross_entropy/softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
]
cross_entropy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
cross_entropy/MeanMean<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2cross_entropy/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
loss_optimizer/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
g
"loss_optimizer/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
loss_optimizer/gradients/FillFillloss_optimizer/gradients/Shape"loss_optimizer/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
>loss_optimizer/gradients/cross_entropy/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/ReshapeReshapeloss_optimizer/gradients/Fill>loss_optimizer/gradients/cross_entropy/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
6loss_optimizer/gradients/cross_entropy/Mean_grad/ShapeShape<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
�
5loss_optimizer/gradients/cross_entropy/Mean_grad/TileTile8loss_optimizer/gradients/cross_entropy/Mean_grad/Reshape6loss_optimizer/gradients/cross_entropy/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_1Shape<cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
{
8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
6loss_optimizer/gradients/cross_entropy/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
5loss_optimizer/gradients/cross_entropy/Mean_grad/ProdProd8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_16loss_optimizer/gradients/cross_entropy/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
7loss_optimizer/gradients/cross_entropy/Mean_grad/Prod_1Prod8loss_optimizer/gradients/cross_entropy/Mean_grad/Shape_28loss_optimizer/gradients/cross_entropy/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
:loss_optimizer/gradients/cross_entropy/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/MaximumMaximum7loss_optimizer/gradients/cross_entropy/Mean_grad/Prod_1:loss_optimizer/gradients/cross_entropy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
9loss_optimizer/gradients/cross_entropy/Mean_grad/floordivFloorDiv5loss_optimizer/gradients/cross_entropy/Mean_grad/Prod8loss_optimizer/gradients/cross_entropy/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
5loss_optimizer/gradients/cross_entropy/Mean_grad/CastCast9loss_optimizer/gradients/cross_entropy/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
8loss_optimizer/gradients/cross_entropy/Mean_grad/truedivRealDiv5loss_optimizer/gradients/cross_entropy/Mean_grad/Tile5loss_optimizer/gradients/cross_entropy/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape2cross_entropy/softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0
�
bloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape8loss_optimizer/gradients/cross_entropy/Mean_grad/truediv`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
#loss_optimizer/gradients/zeros_like	ZerosLike4cross_entropy/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
_loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsbloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshape_loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Tloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mulMul[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims4cross_entropy/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax:cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:������������������
�
Tloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/NegNeg[loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:������������������
�
aloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsbloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshapealoss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
Vloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul]loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Tloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:������������������
�
aloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpU^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mulW^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1
�
iloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityTloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mulb^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:������������������
�
kloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityVloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1b^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*i
_class_
][loc:@loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:������������������
�
^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
�
`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeiloss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency^loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

m
'loss_optimizer/gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
s
)loss_optimizer/gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:

�
7loss_optimizer/gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgs'loss_optimizer/gradients/add_grad/Shape)loss_optimizer/gradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%loss_optimizer/gradients/add_grad/SumSum`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape7loss_optimizer/gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)loss_optimizer/gradients/add_grad/ReshapeReshape%loss_optimizer/gradients/add_grad/Sum'loss_optimizer/gradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
'loss_optimizer/gradients/add_grad/Sum_1Sum`loss_optimizer/gradients/cross_entropy/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape9loss_optimizer/gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
+loss_optimizer/gradients/add_grad/Reshape_1Reshape'loss_optimizer/gradients/add_grad/Sum_1)loss_optimizer/gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
2loss_optimizer/gradients/add_grad/tuple/group_depsNoOp*^loss_optimizer/gradients/add_grad/Reshape,^loss_optimizer/gradients/add_grad/Reshape_1
�
:loss_optimizer/gradients/add_grad/tuple/control_dependencyIdentity)loss_optimizer/gradients/add_grad/Reshape3^loss_optimizer/gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*<
_class2
0.loc:@loss_optimizer/gradients/add_grad/Reshape
�
<loss_optimizer/gradients/add_grad/tuple/control_dependency_1Identity+loss_optimizer/gradients/add_grad/Reshape_13^loss_optimizer/gradients/add_grad/tuple/group_deps*
_output_shapes
:
*
T0*>
_class4
20loc:@loss_optimizer/gradients/add_grad/Reshape_1
�
+loss_optimizer/gradients/MatMul_grad/MatMulMatMul:loss_optimizer/gradients/add_grad/tuple/control_dependencyReadout/weight/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
-loss_optimizer/gradients/MatMul_grad/MatMul_1MatMuldropout/mul:loss_optimizer/gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	�
*
transpose_a(*
transpose_b( 
�
5loss_optimizer/gradients/MatMul_grad/tuple/group_depsNoOp,^loss_optimizer/gradients/MatMul_grad/MatMul.^loss_optimizer/gradients/MatMul_grad/MatMul_1
�
=loss_optimizer/gradients/MatMul_grad/tuple/control_dependencyIdentity+loss_optimizer/gradients/MatMul_grad/MatMul6^loss_optimizer/gradients/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@loss_optimizer/gradients/MatMul_grad/MatMul*(
_output_shapes
:����������
�
?loss_optimizer/gradients/MatMul_grad/tuple/control_dependency_1Identity-loss_optimizer/gradients/MatMul_grad/MatMul_16^loss_optimizer/gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
*
T0*@
_class6
42loc:@loss_optimizer/gradients/MatMul_grad/MatMul_1
�
/loss_optimizer/gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0*#
_output_shapes
:���������
�
1loss_optimizer/gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
_output_shapes
:���������
�
?loss_optimizer/gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs/loss_optimizer/gradients/dropout/mul_grad/Shape1loss_optimizer/gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
-loss_optimizer/gradients/dropout/mul_grad/MulMul=loss_optimizer/gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
�
-loss_optimizer/gradients/dropout/mul_grad/SumSum-loss_optimizer/gradients/dropout/mul_grad/Mul?loss_optimizer/gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1loss_optimizer/gradients/dropout/mul_grad/ReshapeReshape-loss_optimizer/gradients/dropout/mul_grad/Sum/loss_optimizer/gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
/loss_optimizer/gradients/dropout/mul_grad/Mul_1Muldropout/div=loss_optimizer/gradients/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
/loss_optimizer/gradients/dropout/mul_grad/Sum_1Sum/loss_optimizer/gradients/dropout/mul_grad/Mul_1Aloss_optimizer/gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3loss_optimizer/gradients/dropout/mul_grad/Reshape_1Reshape/loss_optimizer/gradients/dropout/mul_grad/Sum_11loss_optimizer/gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
:loss_optimizer/gradients/dropout/mul_grad/tuple/group_depsNoOp2^loss_optimizer/gradients/dropout/mul_grad/Reshape4^loss_optimizer/gradients/dropout/mul_grad/Reshape_1
�
Bloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependencyIdentity1loss_optimizer/gradients/dropout/mul_grad/Reshape;^loss_optimizer/gradients/dropout/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@loss_optimizer/gradients/dropout/mul_grad/Reshape*
_output_shapes
:
�
Dloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependency_1Identity3loss_optimizer/gradients/dropout/mul_grad/Reshape_1;^loss_optimizer/gradients/dropout/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@loss_optimizer/gradients/dropout/mul_grad/Reshape_1*
_output_shapes
:
v
/loss_optimizer/gradients/dropout/div_grad/ShapeShapeFC/relu*
_output_shapes
:*
T0*
out_type0
�
1loss_optimizer/gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:���������
�
?loss_optimizer/gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs/loss_optimizer/gradients/dropout/div_grad/Shape1loss_optimizer/gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1loss_optimizer/gradients/dropout/div_grad/RealDivRealDivBloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
_output_shapes
:*
T0
�
-loss_optimizer/gradients/dropout/div_grad/SumSum1loss_optimizer/gradients/dropout/div_grad/RealDiv?loss_optimizer/gradients/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
1loss_optimizer/gradients/dropout/div_grad/ReshapeReshape-loss_optimizer/gradients/dropout/div_grad/Sum/loss_optimizer/gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
p
-loss_optimizer/gradients/dropout/div_grad/NegNegFC/relu*
T0*(
_output_shapes
:����������
�
3loss_optimizer/gradients/dropout/div_grad/RealDiv_1RealDiv-loss_optimizer/gradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:
�
3loss_optimizer/gradients/dropout/div_grad/RealDiv_2RealDiv3loss_optimizer/gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
�
-loss_optimizer/gradients/dropout/div_grad/mulMulBloss_optimizer/gradients/dropout/mul_grad/tuple/control_dependency3loss_optimizer/gradients/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
/loss_optimizer/gradients/dropout/div_grad/Sum_1Sum-loss_optimizer/gradients/dropout/div_grad/mulAloss_optimizer/gradients/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3loss_optimizer/gradients/dropout/div_grad/Reshape_1Reshape/loss_optimizer/gradients/dropout/div_grad/Sum_11loss_optimizer/gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
:loss_optimizer/gradients/dropout/div_grad/tuple/group_depsNoOp2^loss_optimizer/gradients/dropout/div_grad/Reshape4^loss_optimizer/gradients/dropout/div_grad/Reshape_1
�
Bloss_optimizer/gradients/dropout/div_grad/tuple/control_dependencyIdentity1loss_optimizer/gradients/dropout/div_grad/Reshape;^loss_optimizer/gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*D
_class:
86loc:@loss_optimizer/gradients/dropout/div_grad/Reshape
�
Dloss_optimizer/gradients/dropout/div_grad/tuple/control_dependency_1Identity3loss_optimizer/gradients/dropout/div_grad/Reshape_1;^loss_optimizer/gradients/dropout/div_grad/tuple/group_deps*
T0*F
_class<
:8loc:@loss_optimizer/gradients/dropout/div_grad/Reshape_1*
_output_shapes
:
�
.loss_optimizer/gradients/FC/relu_grad/ReluGradReluGradBloss_optimizer/gradients/dropout/div_grad/tuple/control_dependencyFC/relu*(
_output_shapes
:����������*
T0
s
*loss_optimizer/gradients/FC/add_grad/ShapeShape	FC/MatMul*
_output_shapes
:*
T0*
out_type0
w
,loss_optimizer/gradients/FC/add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
:loss_optimizer/gradients/FC/add_grad/BroadcastGradientArgsBroadcastGradientArgs*loss_optimizer/gradients/FC/add_grad/Shape,loss_optimizer/gradients/FC/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
(loss_optimizer/gradients/FC/add_grad/SumSum.loss_optimizer/gradients/FC/relu_grad/ReluGrad:loss_optimizer/gradients/FC/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,loss_optimizer/gradients/FC/add_grad/ReshapeReshape(loss_optimizer/gradients/FC/add_grad/Sum*loss_optimizer/gradients/FC/add_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
*loss_optimizer/gradients/FC/add_grad/Sum_1Sum.loss_optimizer/gradients/FC/relu_grad/ReluGrad<loss_optimizer/gradients/FC/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
.loss_optimizer/gradients/FC/add_grad/Reshape_1Reshape*loss_optimizer/gradients/FC/add_grad/Sum_1,loss_optimizer/gradients/FC/add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
5loss_optimizer/gradients/FC/add_grad/tuple/group_depsNoOp-^loss_optimizer/gradients/FC/add_grad/Reshape/^loss_optimizer/gradients/FC/add_grad/Reshape_1
�
=loss_optimizer/gradients/FC/add_grad/tuple/control_dependencyIdentity,loss_optimizer/gradients/FC/add_grad/Reshape6^loss_optimizer/gradients/FC/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@loss_optimizer/gradients/FC/add_grad/Reshape*(
_output_shapes
:����������
�
?loss_optimizer/gradients/FC/add_grad/tuple/control_dependency_1Identity.loss_optimizer/gradients/FC/add_grad/Reshape_16^loss_optimizer/gradients/FC/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss_optimizer/gradients/FC/add_grad/Reshape_1*
_output_shapes	
:�
�
.loss_optimizer/gradients/FC/MatMul_grad/MatMulMatMul=loss_optimizer/gradients/FC/add_grad/tuple/control_dependencyFC/weight/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
0loss_optimizer/gradients/FC/MatMul_grad/MatMul_1MatMul
FC/Reshape=loss_optimizer/gradients/FC/add_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
8loss_optimizer/gradients/FC/MatMul_grad/tuple/group_depsNoOp/^loss_optimizer/gradients/FC/MatMul_grad/MatMul1^loss_optimizer/gradients/FC/MatMul_grad/MatMul_1
�
@loss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependencyIdentity.loss_optimizer/gradients/FC/MatMul_grad/MatMul9^loss_optimizer/gradients/FC/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss_optimizer/gradients/FC/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Bloss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependency_1Identity0loss_optimizer/gradients/FC/MatMul_grad/MatMul_19^loss_optimizer/gradients/FC/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@loss_optimizer/gradients/FC/MatMul_grad/MatMul_1* 
_output_shapes
:
��
x
.loss_optimizer/gradients/FC/Reshape_grad/ShapeShape
Conv2/pool*
_output_shapes
:*
T0*
out_type0
�
0loss_optimizer/gradients/FC/Reshape_grad/ReshapeReshape@loss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependency.loss_optimizer/gradients/FC/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
4loss_optimizer/gradients/Conv2/pool_grad/MaxPoolGradMaxPoolGrad
Conv2/relu
Conv2/pool0loss_optimizer/gradients/FC/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
1loss_optimizer/gradients/Conv2/relu_grad/ReluGradReluGrad4loss_optimizer/gradients/Conv2/pool_grad/MaxPoolGrad
Conv2/relu*
T0*/
_output_shapes
:���������@
y
-loss_optimizer/gradients/Conv2/add_grad/ShapeShapeConv2/conv2d*
T0*
out_type0*
_output_shapes
:
y
/loss_optimizer/gradients/Conv2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
�
=loss_optimizer/gradients/Conv2/add_grad/BroadcastGradientArgsBroadcastGradientArgs-loss_optimizer/gradients/Conv2/add_grad/Shape/loss_optimizer/gradients/Conv2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
+loss_optimizer/gradients/Conv2/add_grad/SumSum1loss_optimizer/gradients/Conv2/relu_grad/ReluGrad=loss_optimizer/gradients/Conv2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
/loss_optimizer/gradients/Conv2/add_grad/ReshapeReshape+loss_optimizer/gradients/Conv2/add_grad/Sum-loss_optimizer/gradients/Conv2/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
-loss_optimizer/gradients/Conv2/add_grad/Sum_1Sum1loss_optimizer/gradients/Conv2/relu_grad/ReluGrad?loss_optimizer/gradients/Conv2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1loss_optimizer/gradients/Conv2/add_grad/Reshape_1Reshape-loss_optimizer/gradients/Conv2/add_grad/Sum_1/loss_optimizer/gradients/Conv2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
8loss_optimizer/gradients/Conv2/add_grad/tuple/group_depsNoOp0^loss_optimizer/gradients/Conv2/add_grad/Reshape2^loss_optimizer/gradients/Conv2/add_grad/Reshape_1
�
@loss_optimizer/gradients/Conv2/add_grad/tuple/control_dependencyIdentity/loss_optimizer/gradients/Conv2/add_grad/Reshape9^loss_optimizer/gradients/Conv2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@loss_optimizer/gradients/Conv2/add_grad/Reshape*/
_output_shapes
:���������@
�
Bloss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency_1Identity1loss_optimizer/gradients/Conv2/add_grad/Reshape_19^loss_optimizer/gradients/Conv2/add_grad/tuple/group_deps*
_output_shapes
:@*
T0*D
_class:
86loc:@loss_optimizer/gradients/Conv2/add_grad/Reshape_1
�
1loss_optimizer/gradients/Conv2/conv2d_grad/ShapeNShapeN
Conv1/poolConv2/weights/weight/read*
T0*
out_type0*
N* 
_output_shapes
::
�
>loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput1loss_optimizer/gradients/Conv2/conv2d_grad/ShapeNConv2/weights/weight/read@loss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:��������� *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
?loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilter
Conv1/pool3loss_optimizer/gradients/Conv2/conv2d_grad/ShapeN:1@loss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
;loss_optimizer/gradients/Conv2/conv2d_grad/tuple/group_depsNoOp@^loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilter?^loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInput
�
Closs_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependencyIdentity>loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInput<^loss_optimizer/gradients/Conv2/conv2d_grad/tuple/group_deps*/
_output_shapes
:��������� *
T0*Q
_classG
ECloc:@loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropInput
�
Eloss_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependency_1Identity?loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilter<^loss_optimizer/gradients/Conv2/conv2d_grad/tuple/group_deps*&
_output_shapes
: @*
T0*R
_classH
FDloc:@loss_optimizer/gradients/Conv2/conv2d_grad/Conv2DBackpropFilter
�
4loss_optimizer/gradients/Conv1/pool_grad/MaxPoolGradMaxPoolGrad
Conv1/relu
Conv1/poolCloss_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependency*/
_output_shapes
:��������� *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
1loss_optimizer/gradients/Conv1/relu_grad/ReluGradReluGrad4loss_optimizer/gradients/Conv1/pool_grad/MaxPoolGrad
Conv1/relu*
T0*/
_output_shapes
:��������� 
y
-loss_optimizer/gradients/Conv1/add_grad/ShapeShapeConv1/conv2d*
_output_shapes
:*
T0*
out_type0
y
/loss_optimizer/gradients/Conv1/add_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:
�
=loss_optimizer/gradients/Conv1/add_grad/BroadcastGradientArgsBroadcastGradientArgs-loss_optimizer/gradients/Conv1/add_grad/Shape/loss_optimizer/gradients/Conv1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+loss_optimizer/gradients/Conv1/add_grad/SumSum1loss_optimizer/gradients/Conv1/relu_grad/ReluGrad=loss_optimizer/gradients/Conv1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
/loss_optimizer/gradients/Conv1/add_grad/ReshapeReshape+loss_optimizer/gradients/Conv1/add_grad/Sum-loss_optimizer/gradients/Conv1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
-loss_optimizer/gradients/Conv1/add_grad/Sum_1Sum1loss_optimizer/gradients/Conv1/relu_grad/ReluGrad?loss_optimizer/gradients/Conv1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
1loss_optimizer/gradients/Conv1/add_grad/Reshape_1Reshape-loss_optimizer/gradients/Conv1/add_grad/Sum_1/loss_optimizer/gradients/Conv1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
8loss_optimizer/gradients/Conv1/add_grad/tuple/group_depsNoOp0^loss_optimizer/gradients/Conv1/add_grad/Reshape2^loss_optimizer/gradients/Conv1/add_grad/Reshape_1
�
@loss_optimizer/gradients/Conv1/add_grad/tuple/control_dependencyIdentity/loss_optimizer/gradients/Conv1/add_grad/Reshape9^loss_optimizer/gradients/Conv1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@loss_optimizer/gradients/Conv1/add_grad/Reshape*/
_output_shapes
:��������� 
�
Bloss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency_1Identity1loss_optimizer/gradients/Conv1/add_grad/Reshape_19^loss_optimizer/gradients/Conv1/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@loss_optimizer/gradients/Conv1/add_grad/Reshape_1*
_output_shapes
: 
�
1loss_optimizer/gradients/Conv1/conv2d_grad/ShapeNShapeNInput_Reshape/x_imageConv1/weights/weight/read*
T0*
out_type0*
N* 
_output_shapes
::
�
>loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput1loss_optimizer/gradients/Conv1/conv2d_grad/ShapeNConv1/weights/weight/read@loss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
	dilations

�
?loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterInput_Reshape/x_image3loss_optimizer/gradients/Conv1/conv2d_grad/ShapeN:1@loss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
;loss_optimizer/gradients/Conv1/conv2d_grad/tuple/group_depsNoOp@^loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilter?^loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInput
�
Closs_optimizer/gradients/Conv1/conv2d_grad/tuple/control_dependencyIdentity>loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInput<^loss_optimizer/gradients/Conv1/conv2d_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*Q
_classG
ECloc:@loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropInput
�
Eloss_optimizer/gradients/Conv1/conv2d_grad/tuple/control_dependency_1Identity?loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilter<^loss_optimizer/gradients/Conv1/conv2d_grad/tuple/group_deps*
T0*R
_classH
FDloc:@loss_optimizer/gradients/Conv1/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
(loss_optimizer/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *$
_class
loc:@Conv1/biases/bias*
valueB
 *fff?
�
loss_optimizer/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@Conv1/biases/bias*
	container *
shape: 
�
!loss_optimizer/beta1_power/AssignAssignloss_optimizer/beta1_power(loss_optimizer/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias
�
loss_optimizer/beta1_power/readIdentityloss_optimizer/beta1_power*
_output_shapes
: *
T0*$
_class
loc:@Conv1/biases/bias
�
(loss_optimizer/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *$
_class
loc:@Conv1/biases/bias*
valueB
 *w�?
�
loss_optimizer/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@Conv1/biases/bias*
	container *
shape: 
�
!loss_optimizer/beta2_power/AssignAssignloss_optimizer/beta2_power(loss_optimizer/beta2_power/initial_value*
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
loss_optimizer/beta2_power/readIdentityloss_optimizer/beta2_power*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
+Conv1/weights/weight/Adam/Initializer/zerosConst*'
_class
loc:@Conv1/weights/weight*%
valueB *    *
dtype0*&
_output_shapes
: 
�
Conv1/weights/weight/Adam
VariableV2*
shared_name *'
_class
loc:@Conv1/weights/weight*
	container *
shape: *
dtype0*&
_output_shapes
: 
�
 Conv1/weights/weight/Adam/AssignAssignConv1/weights/weight/Adam+Conv1/weights/weight/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Conv1/weights/weight
�
Conv1/weights/weight/Adam/readIdentityConv1/weights/weight/Adam*
T0*'
_class
loc:@Conv1/weights/weight*&
_output_shapes
: 
�
-Conv1/weights/weight/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
: *'
_class
loc:@Conv1/weights/weight*%
valueB *    
�
Conv1/weights/weight/Adam_1
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *'
_class
loc:@Conv1/weights/weight*
	container *
shape: 
�
"Conv1/weights/weight/Adam_1/AssignAssignConv1/weights/weight/Adam_1-Conv1/weights/weight/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Conv1/weights/weight*
validate_shape(*&
_output_shapes
: 
�
 Conv1/weights/weight/Adam_1/readIdentityConv1/weights/weight/Adam_1*
T0*'
_class
loc:@Conv1/weights/weight*&
_output_shapes
: 
�
(Conv1/biases/bias/Adam/Initializer/zerosConst*$
_class
loc:@Conv1/biases/bias*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv1/biases/bias/Adam
VariableV2*
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@Conv1/biases/bias*
	container *
shape: 
�
Conv1/biases/bias/Adam/AssignAssignConv1/biases/bias/Adam(Conv1/biases/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@Conv1/biases/bias
�
Conv1/biases/bias/Adam/readIdentityConv1/biases/bias/Adam*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
*Conv1/biases/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@Conv1/biases/bias*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv1/biases/bias/Adam_1
VariableV2*$
_class
loc:@Conv1/biases/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Conv1/biases/bias/Adam_1/AssignAssignConv1/biases/bias/Adam_1*Conv1/biases/bias/Adam_1/Initializer/zeros*
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
Conv1/biases/bias/Adam_1/readIdentityConv1/biases/bias/Adam_1*
_output_shapes
: *
T0*$
_class
loc:@Conv1/biases/bias
�
;Conv2/weights/weight/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Conv2/weights/weight*%
valueB"          @   *
dtype0*
_output_shapes
:
�
1Conv2/weights/weight/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@Conv2/weights/weight*
valueB
 *    
�
+Conv2/weights/weight/Adam/Initializer/zerosFill;Conv2/weights/weight/Adam/Initializer/zeros/shape_as_tensor1Conv2/weights/weight/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@Conv2/weights/weight*

index_type0*&
_output_shapes
: @
�
Conv2/weights/weight/Adam
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *'
_class
loc:@Conv2/weights/weight*
	container *
shape: @
�
 Conv2/weights/weight/Adam/AssignAssignConv2/weights/weight/Adam+Conv2/weights/weight/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Conv2/weights/weight*
validate_shape(*&
_output_shapes
: @
�
Conv2/weights/weight/Adam/readIdentityConv2/weights/weight/Adam*&
_output_shapes
: @*
T0*'
_class
loc:@Conv2/weights/weight
�
=Conv2/weights/weight/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Conv2/weights/weight*%
valueB"          @   *
dtype0*
_output_shapes
:
�
3Conv2/weights/weight/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@Conv2/weights/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-Conv2/weights/weight/Adam_1/Initializer/zerosFill=Conv2/weights/weight/Adam_1/Initializer/zeros/shape_as_tensor3Conv2/weights/weight/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@Conv2/weights/weight*

index_type0*&
_output_shapes
: @
�
Conv2/weights/weight/Adam_1
VariableV2*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *'
_class
loc:@Conv2/weights/weight
�
"Conv2/weights/weight/Adam_1/AssignAssignConv2/weights/weight/Adam_1-Conv2/weights/weight/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*'
_class
loc:@Conv2/weights/weight
�
 Conv2/weights/weight/Adam_1/readIdentityConv2/weights/weight/Adam_1*
T0*'
_class
loc:@Conv2/weights/weight*&
_output_shapes
: @
�
(Conv2/biases/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*$
_class
loc:@Conv2/biases/bias*
valueB@*    
�
Conv2/biases/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *$
_class
loc:@Conv2/biases/bias*
	container *
shape:@
�
Conv2/biases/bias/Adam/AssignAssignConv2/biases/bias/Adam(Conv2/biases/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@Conv2/biases/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
Conv2/biases/bias/Adam/readIdentityConv2/biases/bias/Adam*
_output_shapes
:@*
T0*$
_class
loc:@Conv2/biases/bias
�
*Conv2/biases/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@Conv2/biases/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2/biases/bias/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *$
_class
loc:@Conv2/biases/bias
�
Conv2/biases/bias/Adam_1/AssignAssignConv2/biases/bias/Adam_1*Conv2/biases/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Conv2/biases/bias*
validate_shape(*
_output_shapes
:@
�
Conv2/biases/bias/Adam_1/readIdentityConv2/biases/bias/Adam_1*
T0*$
_class
loc:@Conv2/biases/bias*
_output_shapes
:@
�
0FC/weight/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@FC/weight*
valueB"@     
�
&FC/weight/Adam/Initializer/zeros/ConstConst*
_class
loc:@FC/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 FC/weight/Adam/Initializer/zerosFill0FC/weight/Adam/Initializer/zeros/shape_as_tensor&FC/weight/Adam/Initializer/zeros/Const*
T0*
_class
loc:@FC/weight*

index_type0* 
_output_shapes
:
��
�
FC/weight/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *
_class
loc:@FC/weight*
	container *
shape:
��
�
FC/weight/Adam/AssignAssignFC/weight/Adam FC/weight/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/weight*
validate_shape(* 
_output_shapes
:
��
x
FC/weight/Adam/readIdentityFC/weight/Adam* 
_output_shapes
:
��*
T0*
_class
loc:@FC/weight
�
2FC/weight/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@FC/weight*
valueB"@     *
dtype0*
_output_shapes
:
�
(FC/weight/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@FC/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"FC/weight/Adam_1/Initializer/zerosFill2FC/weight/Adam_1/Initializer/zeros/shape_as_tensor(FC/weight/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@FC/weight*

index_type0* 
_output_shapes
:
��
�
FC/weight/Adam_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *
_class
loc:@FC/weight*
	container 
�
FC/weight/Adam_1/AssignAssignFC/weight/Adam_1"FC/weight/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*
_class
loc:@FC/weight
|
FC/weight/Adam_1/readIdentityFC/weight/Adam_1*
T0*
_class
loc:@FC/weight* 
_output_shapes
:
��
�
.FC/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@FC/bias*
valueB:�
�
$FC/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@FC/bias*
valueB
 *    
�
FC/bias/Adam/Initializer/zerosFill.FC/bias/Adam/Initializer/zeros/shape_as_tensor$FC/bias/Adam/Initializer/zeros/Const*
T0*
_class
loc:@FC/bias*

index_type0*
_output_shapes	
:�
�
FC/bias/Adam
VariableV2*
shared_name *
_class
loc:@FC/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
FC/bias/Adam/AssignAssignFC/bias/AdamFC/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/bias*
validate_shape(*
_output_shapes	
:�
m
FC/bias/Adam/readIdentityFC/bias/Adam*
T0*
_class
loc:@FC/bias*
_output_shapes	
:�
�
0FC/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@FC/bias*
valueB:�
�
&FC/bias/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@FC/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 FC/bias/Adam_1/Initializer/zerosFill0FC/bias/Adam_1/Initializer/zeros/shape_as_tensor&FC/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*
_class
loc:@FC/bias*

index_type0
�
FC/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@FC/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
FC/bias/Adam_1/AssignAssignFC/bias/Adam_1 FC/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@FC/bias*
validate_shape(*
_output_shapes	
:�
q
FC/bias/Adam_1/readIdentityFC/bias/Adam_1*
T0*
_class
loc:@FC/bias*
_output_shapes	
:�
�
5Readout/weight/Adam/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@Readout/weight*
valueB"   
   *
dtype0*
_output_shapes
:
�
+Readout/weight/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *!
_class
loc:@Readout/weight*
valueB
 *    
�
%Readout/weight/Adam/Initializer/zerosFill5Readout/weight/Adam/Initializer/zeros/shape_as_tensor+Readout/weight/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@Readout/weight*

index_type0*
_output_shapes
:	�

�
Readout/weight/Adam
VariableV2*!
_class
loc:@Readout/weight*
	container *
shape:	�
*
dtype0*
_output_shapes
:	�
*
shared_name 
�
Readout/weight/Adam/AssignAssignReadout/weight/Adam%Readout/weight/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�
*
use_locking(*
T0*!
_class
loc:@Readout/weight
�
Readout/weight/Adam/readIdentityReadout/weight/Adam*
_output_shapes
:	�
*
T0*!
_class
loc:@Readout/weight
�
7Readout/weight/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@Readout/weight*
valueB"   
   *
dtype0*
_output_shapes
:
�
-Readout/weight/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@Readout/weight*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'Readout/weight/Adam_1/Initializer/zerosFill7Readout/weight/Adam_1/Initializer/zeros/shape_as_tensor-Readout/weight/Adam_1/Initializer/zeros/Const*
_output_shapes
:	�
*
T0*!
_class
loc:@Readout/weight*

index_type0
�
Readout/weight/Adam_1
VariableV2*
shape:	�
*
dtype0*
_output_shapes
:	�
*
shared_name *!
_class
loc:@Readout/weight*
	container 
�
Readout/weight/Adam_1/AssignAssignReadout/weight/Adam_1'Readout/weight/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Readout/weight*
validate_shape(*
_output_shapes
:	�

�
Readout/weight/Adam_1/readIdentityReadout/weight/Adam_1*
T0*!
_class
loc:@Readout/weight*
_output_shapes
:	�

�
#Readout/bias/Adam/Initializer/zerosConst*
_class
loc:@Readout/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
Readout/bias/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@Readout/bias*
	container *
shape:

�
Readout/bias/Adam/AssignAssignReadout/bias/Adam#Readout/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Readout/bias*
validate_shape(*
_output_shapes
:

{
Readout/bias/Adam/readIdentityReadout/bias/Adam*
T0*
_class
loc:@Readout/bias*
_output_shapes
:

�
%Readout/bias/Adam_1/Initializer/zerosConst*
_class
loc:@Readout/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
Readout/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@Readout/bias*
	container *
shape:
*
dtype0*
_output_shapes
:

�
Readout/bias/Adam_1/AssignAssignReadout/bias/Adam_1%Readout/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Readout/bias*
validate_shape(*
_output_shapes
:


Readout/bias/Adam_1/readIdentityReadout/bias/Adam_1*
_output_shapes
:
*
T0*
_class
loc:@Readout/bias
f
!loss_optimizer/Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
^
loss_optimizer/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
^
loss_optimizer/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
`
loss_optimizer/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
9loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam	ApplyAdamConv1/weights/weightConv1/weights/weight/AdamConv1/weights/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonEloss_optimizer/gradients/Conv1/conv2d_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@Conv1/weights/weight*
use_nesterov( *&
_output_shapes
: 
�
6loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam	ApplyAdamConv1/biases/biasConv1/biases/bias/AdamConv1/biases/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonBloss_optimizer/gradients/Conv1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Conv1/biases/bias*
use_nesterov( *
_output_shapes
: 
�
9loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam	ApplyAdamConv2/weights/weightConv2/weights/weight/AdamConv2/weights/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonEloss_optimizer/gradients/Conv2/conv2d_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
: @*
use_locking( *
T0*'
_class
loc:@Conv2/weights/weight
�
6loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam	ApplyAdamConv2/biases/biasConv2/biases/bias/AdamConv2/biases/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonBloss_optimizer/gradients/Conv2/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*$
_class
loc:@Conv2/biases/bias
�
.loss_optimizer/Adam/update_FC/weight/ApplyAdam	ApplyAdam	FC/weightFC/weight/AdamFC/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilonBloss_optimizer/gradients/FC/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@FC/weight*
use_nesterov( * 
_output_shapes
:
��
�
,loss_optimizer/Adam/update_FC/bias/ApplyAdam	ApplyAdamFC/biasFC/bias/AdamFC/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilon?loss_optimizer/gradients/FC/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@FC/bias*
use_nesterov( *
_output_shapes	
:�
�
3loss_optimizer/Adam/update_Readout/weight/ApplyAdam	ApplyAdamReadout/weightReadout/weight/AdamReadout/weight/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilon?loss_optimizer/gradients/MatMul_grad/tuple/control_dependency_1*
T0*!
_class
loc:@Readout/weight*
use_nesterov( *
_output_shapes
:	�
*
use_locking( 
�
1loss_optimizer/Adam/update_Readout/bias/ApplyAdam	ApplyAdamReadout/biasReadout/bias/AdamReadout/bias/Adam_1loss_optimizer/beta1_power/readloss_optimizer/beta2_power/read!loss_optimizer/Adam/learning_rateloss_optimizer/Adam/beta1loss_optimizer/Adam/beta2loss_optimizer/Adam/epsilon<loss_optimizer/gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Readout/bias
�
loss_optimizer/Adam/mulMulloss_optimizer/beta1_power/readloss_optimizer/Adam/beta17^loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam7^loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam-^loss_optimizer/Adam/update_FC/bias/ApplyAdam/^loss_optimizer/Adam/update_FC/weight/ApplyAdam2^loss_optimizer/Adam/update_Readout/bias/ApplyAdam4^loss_optimizer/Adam/update_Readout/weight/ApplyAdam*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
loss_optimizer/Adam/AssignAssignloss_optimizer/beta1_powerloss_optimizer/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*$
_class
loc:@Conv1/biases/bias
�
loss_optimizer/Adam/mul_1Mulloss_optimizer/beta2_power/readloss_optimizer/Adam/beta27^loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam7^loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam-^loss_optimizer/Adam/update_FC/bias/ApplyAdam/^loss_optimizer/Adam/update_FC/weight/ApplyAdam2^loss_optimizer/Adam/update_Readout/bias/ApplyAdam4^loss_optimizer/Adam/update_Readout/weight/ApplyAdam*
T0*$
_class
loc:@Conv1/biases/bias*
_output_shapes
: 
�
loss_optimizer/Adam/Assign_1Assignloss_optimizer/beta2_powerloss_optimizer/Adam/mul_1*
use_locking( *
T0*$
_class
loc:@Conv1/biases/bias*
validate_shape(*
_output_shapes
: 
�
loss_optimizer/AdamNoOp^loss_optimizer/Adam/Assign^loss_optimizer/Adam/Assign_17^loss_optimizer/Adam/update_Conv1/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv1/weights/weight/ApplyAdam7^loss_optimizer/Adam/update_Conv2/biases/bias/ApplyAdam:^loss_optimizer/Adam/update_Conv2/weights/weight/ApplyAdam-^loss_optimizer/Adam/update_FC/bias/ApplyAdam/^loss_optimizer/Adam/update_FC/weight/ApplyAdam2^loss_optimizer/Adam/update_Readout/bias/ApplyAdam4^loss_optimizer/Adam/update_Readout/weight/ApplyAdam
[
accuracy/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMaxArgMaxaddaccuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
]
accuracy/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
accuracy/ArgMax_1ArgMaxMNIST_Input/y_accuracy/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
i
accuracy/EqualEqualaccuracy/ArgMaxaccuracy/ArgMax_1*#
_output_shapes
:���������*
T0	
r
accuracy/CastCastaccuracy/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
X
accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
accuracy/MeanMeanaccuracy/Castaccuracy/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
h
cross_entropy_scl/tagsConst*"
valueB Bcross_entropy_scl*
dtype0*
_output_shapes
: 
o
cross_entropy_sclScalarSummarycross_entropy_scl/tagscross_entropy/Mean*
_output_shapes
: *
T0
h
training_accuracy/tagsConst*
dtype0*
_output_shapes
: *"
valueB Btraining_accuracy
j
training_accuracyScalarSummarytraining_accuracy/tagsaccuracy/Mean*
_output_shapes
: *
T0
�
Merge/MergeSummaryMergeSummaryInput_Reshape/input_imgConv1/weights/summaries/mean_1 Conv1/weights/summaries/stddev_1Conv1/weights/summaries/max_1Conv1/weights/summaries/min_1!Conv1/weights/summaries/histogramConv1/biases/summaries/mean_1Conv1/biases/summaries/stddev_1Conv1/biases/summaries/max_1Conv1/biases/summaries/min_1 Conv1/biases/summaries/histogramConv1/conv1_wx_bConv1/h_conv1Conv2/weights/summaries/mean_1 Conv2/weights/summaries/stddev_1Conv2/weights/summaries/max_1Conv2/weights/summaries/min_1!Conv2/weights/summaries/histogramConv2/biases/summaries/mean_1Conv2/biases/summaries/stddev_1Conv2/biases/summaries/max_1Conv2/biases/summaries/min_1 Conv2/biases/summaries/histogramConv2/conv2_wx_bConv2/h_conv2cross_entropy_scltraining_accuracy*
N*
_output_shapes
: 
�
initNoOp^Conv1/biases/bias/Adam/Assign ^Conv1/biases/bias/Adam_1/Assign^Conv1/biases/bias/Assign!^Conv1/weights/weight/Adam/Assign#^Conv1/weights/weight/Adam_1/Assign^Conv1/weights/weight/Assign^Conv2/biases/bias/Adam/Assign ^Conv2/biases/bias/Adam_1/Assign^Conv2/biases/bias/Assign!^Conv2/weights/weight/Adam/Assign#^Conv2/weights/weight/Adam_1/Assign^Conv2/weights/weight/Assign^FC/bias/Adam/Assign^FC/bias/Adam_1/Assign^FC/bias/Assign^FC/weight/Adam/Assign^FC/weight/Adam_1/Assign^FC/weight/Assign^Readout/bias/Adam/Assign^Readout/bias/Adam_1/Assign^Readout/bias/Assign^Readout/weight/Adam/Assign^Readout/weight/Adam_1/Assign^Readout/weight/Assign"^loss_optimizer/beta1_power/Assign"^loss_optimizer/beta2_power/Assign""�
trainable_variables��
v
Conv1/weights/weight:0Conv1/weights/weight/AssignConv1/weights/weight/read:02 Conv1/weights/truncated_normal:08
a
Conv1/biases/bias:0Conv1/biases/bias/AssignConv1/biases/bias/read:02Conv1/biases/Const:08
v
Conv2/weights/weight:0Conv2/weights/weight/AssignConv2/weights/weight/read:02 Conv2/weights/truncated_normal:08
a
Conv2/biases/bias:0Conv2/biases/bias/AssignConv2/biases/bias/read:02Conv2/biases/Const:08
J
FC/weight:0FC/weight/AssignFC/weight/read:02FC/truncated_normal:08
9
	FC/bias:0FC/bias/AssignFC/bias/read:02
FC/Const:08
^
Readout/weight:0Readout/weight/AssignReadout/weight/read:02Readout/truncated_normal:08
M
Readout/bias:0Readout/bias/AssignReadout/bias/read:02Readout/Const:08"�
	summaries�
�
Input_Reshape/input_img:0
 Conv1/weights/summaries/mean_1:0
"Conv1/weights/summaries/stddev_1:0
Conv1/weights/summaries/max_1:0
Conv1/weights/summaries/min_1:0
#Conv1/weights/summaries/histogram:0
Conv1/biases/summaries/mean_1:0
!Conv1/biases/summaries/stddev_1:0
Conv1/biases/summaries/max_1:0
Conv1/biases/summaries/min_1:0
"Conv1/biases/summaries/histogram:0
Conv1/conv1_wx_b:0
Conv1/h_conv1:0
 Conv2/weights/summaries/mean_1:0
"Conv2/weights/summaries/stddev_1:0
Conv2/weights/summaries/max_1:0
Conv2/weights/summaries/min_1:0
#Conv2/weights/summaries/histogram:0
Conv2/biases/summaries/mean_1:0
!Conv2/biases/summaries/stddev_1:0
Conv2/biases/summaries/max_1:0
Conv2/biases/summaries/min_1:0
"Conv2/biases/summaries/histogram:0
Conv2/conv2_wx_b:0
Conv2/h_conv2:0
cross_entropy_scl:0
training_accuracy:0"#
train_op

loss_optimizer/Adam"�
	variables��
v
Conv1/weights/weight:0Conv1/weights/weight/AssignConv1/weights/weight/read:02 Conv1/weights/truncated_normal:08
a
Conv1/biases/bias:0Conv1/biases/bias/AssignConv1/biases/bias/read:02Conv1/biases/Const:08
v
Conv2/weights/weight:0Conv2/weights/weight/AssignConv2/weights/weight/read:02 Conv2/weights/truncated_normal:08
a
Conv2/biases/bias:0Conv2/biases/bias/AssignConv2/biases/bias/read:02Conv2/biases/Const:08
J
FC/weight:0FC/weight/AssignFC/weight/read:02FC/truncated_normal:08
9
	FC/bias:0FC/bias/AssignFC/bias/read:02
FC/Const:08
^
Readout/weight:0Readout/weight/AssignReadout/weight/read:02Readout/truncated_normal:08
M
Readout/bias:0Readout/bias/AssignReadout/bias/read:02Readout/Const:08
�
loss_optimizer/beta1_power:0!loss_optimizer/beta1_power/Assign!loss_optimizer/beta1_power/read:02*loss_optimizer/beta1_power/initial_value:0
�
loss_optimizer/beta2_power:0!loss_optimizer/beta2_power/Assign!loss_optimizer/beta2_power/read:02*loss_optimizer/beta2_power/initial_value:0
�
Conv1/weights/weight/Adam:0 Conv1/weights/weight/Adam/Assign Conv1/weights/weight/Adam/read:02-Conv1/weights/weight/Adam/Initializer/zeros:0
�
Conv1/weights/weight/Adam_1:0"Conv1/weights/weight/Adam_1/Assign"Conv1/weights/weight/Adam_1/read:02/Conv1/weights/weight/Adam_1/Initializer/zeros:0
�
Conv1/biases/bias/Adam:0Conv1/biases/bias/Adam/AssignConv1/biases/bias/Adam/read:02*Conv1/biases/bias/Adam/Initializer/zeros:0
�
Conv1/biases/bias/Adam_1:0Conv1/biases/bias/Adam_1/AssignConv1/biases/bias/Adam_1/read:02,Conv1/biases/bias/Adam_1/Initializer/zeros:0
�
Conv2/weights/weight/Adam:0 Conv2/weights/weight/Adam/Assign Conv2/weights/weight/Adam/read:02-Conv2/weights/weight/Adam/Initializer/zeros:0
�
Conv2/weights/weight/Adam_1:0"Conv2/weights/weight/Adam_1/Assign"Conv2/weights/weight/Adam_1/read:02/Conv2/weights/weight/Adam_1/Initializer/zeros:0
�
Conv2/biases/bias/Adam:0Conv2/biases/bias/Adam/AssignConv2/biases/bias/Adam/read:02*Conv2/biases/bias/Adam/Initializer/zeros:0
�
Conv2/biases/bias/Adam_1:0Conv2/biases/bias/Adam_1/AssignConv2/biases/bias/Adam_1/read:02,Conv2/biases/bias/Adam_1/Initializer/zeros:0
d
FC/weight/Adam:0FC/weight/Adam/AssignFC/weight/Adam/read:02"FC/weight/Adam/Initializer/zeros:0
l
FC/weight/Adam_1:0FC/weight/Adam_1/AssignFC/weight/Adam_1/read:02$FC/weight/Adam_1/Initializer/zeros:0
\
FC/bias/Adam:0FC/bias/Adam/AssignFC/bias/Adam/read:02 FC/bias/Adam/Initializer/zeros:0
d
FC/bias/Adam_1:0FC/bias/Adam_1/AssignFC/bias/Adam_1/read:02"FC/bias/Adam_1/Initializer/zeros:0
x
Readout/weight/Adam:0Readout/weight/Adam/AssignReadout/weight/Adam/read:02'Readout/weight/Adam/Initializer/zeros:0
�
Readout/weight/Adam_1:0Readout/weight/Adam_1/AssignReadout/weight/Adam_1/read:02)Readout/weight/Adam_1/Initializer/zeros:0
p
Readout/bias/Adam:0Readout/bias/Adam/AssignReadout/bias/Adam/read:02%Readout/bias/Adam/Initializer/zeros:0
x
Readout/bias/Adam_1:0Readout/bias/Adam_1/AssignReadout/bias/Adam_1/read:02'Readout/bias/Adam_1/Initializer/zeros:0)�ʴT      8�	� X���A*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   IDAT(��СK�A��߫�$���`X0�X,���b˻b��^�f�Q�}Ĥ ���^LtozS�_β_�k�t���sw�?+O�4s�p59��n�rd����5!�lƘ�sD���~�{F/���=�M;���ڭ�
�D�M��@n��� �M��t�*Iz�J�w%ݭwG�}ҳ0�͝|Sύp9�L�z�ד�����'k�W�Ģt�J�N!��O���P��y=�)[b��м��H�O\�����g�|0�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H  IDAT(�c` 0c�?c8
e2aHf�]�iT�ӇB8%W�]�S���=%\r��&ᒓ�����g�n��!Ž��[]�W��}k��� �����R4���M������5T���-��:�.~�	E.������K��<�F��z�����}ƿW%����q������a������>�C14��]����d<�R���k������&K.�c|����``(ۆ�	�k?�����_����[�MQ�&9D  ]va���_X    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  ,IDAT(����+�q�_l�YH�eE~������8�XI���99(��Ѯ㠴r0;p�8�%E�]�c��΁-}6W����~^��><��USV�-@�2U1徒$���@�g/��>�S �����.�� �����9�����|�,�P[��D5s�"-�kɞ Lsk�;Vsz�k����tf�Ӗ�aǚ4����ǳ!���6<���/%�Z7���].��z#�=r
��䜰4t����֧0��K�㑄��͗�N(���2!8]���'��K�13
 ���2�?o�9��f�*���HTk��:%Z    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H  IDAT(�c`@`6�ӿ����-fƔ�����߫���ۆ)�2ob���?�8M���$'.I�����8L�r�SX��k�����%)�0lr�����f�E���-v�߿c����{�1%�����������޴Ődξr�R83˽�����������`�30$�q���o�����V�b````1Sex�E�ï�$����ۈj�ǿ�B��y������C�"�w� ��*�J�������j��  ��jі`�    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@�n9�-oP��26���P$��X����WA�|�������A��ɗ��W��,'�738�2�Ϲ�Y8���?����P�
C�;YT��}d���y�*�����Ϥ�ۓ��n!8L貌+qK��?���O�p�Y����C3����$�g|���I��'ɉ[���#�� .=1����    IEND�B`�
%
Conv1/weights/summaries/mean_1w�8�
'
 Conv1/weights/summaries/stddev_1��=
$
Conv1/weights/summaries/max_1OH>
$
Conv1/weights/summaries/min_1�K�
�
!Conv1/weights/summaries/histogram*�	    �tɿ   `�	�?      �@! `�'��)�ؽ@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof��lDZrS�nK���LQ�k�1^�sO�IcD���L�f�ʜ�7
������6�]���nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @      ,@      0@      0@      6@      8@      0@      5@      1@      7@      2@      (@      4@      &@      2@      6@      1@      $@      0@      $@      @      &@      @      @       @      @       @      @      @      @      @       @      @       @       @      �?               @       @      @       @              �?       @      �?              �?              �?              �?      �?              �?      �?              �?      �?               @              �?       @       @      �?      �?       @       @       @      �?      @      @      @      @      @      @      @      @      �?      @      @      "@      *@      &@      @      @      *@      @      &@      *@      2@      6@      &@      5@      2@      ,@      &@      8@      "@      6@      ,@      0@      ,@      .@        
$
Conv1/biases/summaries/mean_1���=
&
Conv1/biases/summaries/stddev_1    
#
Conv1/biases/summaries/max_1���=
#
Conv1/biases/summaries/min_1���=
�
 Conv1/biases/summaries/histogram*a	   ����?   ����?      @@!   ���	@) ��Q�z�?28/�C�ַ?%g�cE9�?�������:              @@        
�"
Conv1/conv1_wx_b*�"	   �j��    ��?     $3A! M�%F��@)���!;��@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`���(��澢f�����uE���⾙ѩ�-߾E��a�WܾK+�E��Ͼ['�?�;�*��ڽ�G&�$��豪}0ڰ�������u��gr��R%�������u`P+d�>0�6�/n�>�����>
�/eq
�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              2@     �e@     (�@     8�@     ��@     $�@     ؞@     �@     ��@     Ƨ@     ت@     ^�@     �@     �@     ��@     ��@     γ@     ��@      �@     ��@     õ@     ��@     �@     q�@     s�@     Ų@     ��@     D�@     �@     ~�@      �@     N�@     ��@     "�@     �@      �@     ��@     ��@     �@     H�@     P�@     ț@     �@     h�@     ��@     ��@     ؑ@     l�@     H�@     `�@     ��@     x�@     ��@     x�@     ��@     X�@     �}@     �z@     pv@     `u@     `u@     pr@     0r@     @l@     `i@     �f@     @g@     @b@     �c@     @b@     @]@      Y@      Y@     �Z@     @W@     �P@      N@      Q@      O@     �H@      G@      A@      >@     �H@      ?@      ?@      >@      6@      ,@      0@      2@      3@      5@      "@      (@      (@      @      $@      &@      @      @       @      $@      @      @      @              @      @      @       @      �?      @      �?               @       @      �?               @               @       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?               @      @              �?      @      @      �?              �?       @       @      @      @      @      @      @      @      @      @      @       @      $@      @      @      @      ,@      "@       @      @      $@      "@      6@      2@      2@      =@      4@      >@      >@     �@@     �B@     �J@     �D@     �D@      M@     �O@      O@     �Q@     @R@     �R@      Y@     @V@     �Z@     �`@     @`@     @d@     �h@      k@     @g@      j@     `o@     @t@     �r@     �u@     �v@     p{@     {@     �y@     Ȁ@     ��@     �@      �@     h�@     ،@     ��@     ��@     D�@     �@     (�@     �@     ��@     D�@     ��@     ԣ@     ��@     ��@     �@     q�@     �@     :�@     ^�@     (�@     .�@     ��@     ��@    ���@    f�%A     ��@    ��@    @�@    ��@     ��@     ��@    ��@     ��@    ���@    � �@    ���@     ��@     ��@    ���@    �Y�@    ���@     �@     T�@     u�@     �@     ��@     ��@     ��@     p�@     @h@      P@      @        
�
Conv1/h_conv1*�    ��?     $3A!��� �A)�o�����@2�        �-���q=�u`P+d�>0�6�/n�>�����>
�/eq
�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            ��A              �?              �?              �?              �?      �?       @      �?               @      @              �?      @      @      �?              �?       @       @      @      @      @      @      @      @      @      @      @       @      $@      @      @      @      ,@      "@       @      @      $@      "@      6@      2@      2@      =@      4@      >@      >@     �@@     �B@     �J@     �D@     �D@      M@     �O@      O@     �Q@     @R@     �R@      Y@     @V@     �Z@     �`@     @`@     @d@     �h@      k@     @g@      j@     `o@     @t@     �r@     �u@     �v@     p{@     {@     �y@     Ȁ@     ��@     �@      �@     h�@     ،@     ��@     ��@     D�@     �@     (�@     �@     ��@     D�@     ��@     ԣ@     ��@     ��@     �@     q�@     �@     :�@     ^�@     (�@     .�@     ��@     ��@    ���@    f�%A     ��@    ��@    @�@    ��@     ��@     ��@    ��@     ��@    ���@    � �@    ���@     ��@     ��@    ���@    �Y�@    ���@     �@     T�@     u�@     �@     ��@     ��@     ��@     p�@     @h@      P@      @        
%
Conv2/weights/summaries/mean_1�:m9
'
 Conv2/weights/summaries/stddev_1X�=
$
Conv2/weights/summaries/max_1'�L>
$
Conv2/weights/summaries/min_1{�L�
�
!Conv2/weights/summaries/histogram*�	   `��ɿ   �$��?      �@!�w��*'@)�߶lH�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�6�]���1��a˲�O�ʗ�����Zr[v��I��P=���iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ
�}���>X$�z�>['�?��>K+�E���>jqs&\��>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              "@     @�@     ��@     ��@     ��@     �@     ��@     ��@     ��@     @�@     ��@     ,�@     �@     ��@     8�@     X�@     h�@     X�@     ��@     ��@     ��@     ��@     @@     |@     �w@     �x@     �u@     `u@      r@     �o@      n@     �o@     �i@     �e@     �b@      b@     @^@     �`@     �^@     @[@     �X@      V@     �S@     @U@      L@     �L@     �K@     �G@     �E@      B@     �E@      @@      3@      7@      6@      ;@      5@      6@      1@      1@       @      &@      @      ,@      $@      $@      @      "@      @      @       @      @      @      @       @      @      @      @      �?      @      �?              @      @      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?      �?       @               @      �?               @      @              �?              �?              �?      �?              �?      @      �?      @      @      @      @      @      @      @      @      @      &@       @      @      @      $@      *@      5@      0@      6@      &@      9@      0@      6@      4@      9@     �@@     �B@      @@     �B@      F@      J@      K@     �R@     �P@     �Q@     �S@     @X@     @X@     @[@     �\@      `@     @c@      d@     @d@     �h@     �g@      k@     �n@     �o@      s@      v@     �t@     0w@     `y@     �z@     P~@     ��@     X�@     �@     (�@     8�@     `�@     ��@     (�@     �@     ܐ@     ��@     0�@     0�@     x�@     L�@     ܒ@     ��@      �@     x�@      �@     ��@       @        
$
Conv2/biases/summaries/mean_1���=
&
Conv2/biases/summaries/stddev_1   2
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1���=
�
 Conv2/biases/summaries/histogram*a	   ����?   ����?      P@!   ���@)��Q�z�?28/�C�ַ?%g�cE9�?�������:              P@        
�$
Conv2/conv2_wx_b*�#	   ��Y�   `@     $#A! ���q�@)_���A2��Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(龢f�����uE�����_�T�l׾��>M|Kվ;�"�qʾ
�/eq
Ⱦ39W$:���.��fc�����|�~�>���]���>��~���>�XQ��>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?       @      ;@     @U@     �m@     �x@     �@     ��@     ��@     ��@     �@     
�@     ��@     6�@     ��@     u�@     ��@     ��@     �@     %�@     �@     ��@     <�@     ��@     w�@     ��@     ��@     ~�@     ܺ@     w�@     ��@     �@     Z�@     `�@     2�@     ��@     5�@     �@     ��@     �@     ,�@     n�@     v�@     ��@     ��@     �@     H�@     �@     `�@     P�@     @�@     ,�@     p�@     Đ@     ��@     ؋@     ��@     ��@     �@     ��@     ��@     `@     z@     �}@     ��@     �u@     �u@      {@     `p@      n@      n@     �l@      d@      r@      _@     @`@     @d@     �`@     �T@      _@     �Z@     �G@      O@      G@      G@     �N@      J@      N@     �@@      B@      8@     �H@      ;@      6@      C@      2@      :@      2@      ,@      &@      (@      .@      @      @      @      @       @      @      @      &@      @      �?      �?       @       @      �?      �?      �?      @       @       @      �?      �?      �?       @      �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?      �?      �?              �?     �^@              �?      �?      �?      @       @      �?      @      @      �?      @      @      @      @       @      @      @       @      @      @      8@       @      "@      @      @      .@      $@      @       @      @      $@      (@      .@       @      .@      1@      6@      .@      5@      8@      6@      =@     �D@      D@     �P@      W@     �J@     �J@     �Q@      M@     �V@     @U@      X@     �U@     �Z@      i@     @_@     �n@     `g@     �c@     �v@     y@     0u@      o@     0r@     @r@     �v@     �@     �{@     �~@     ��@     0�@     ��@     ��@     X�@     �@     `�@     ��@     Ԓ@     ��@     ��@     `�@     ܛ@     h�@     �@     �@     n�@     ��@     R�@     X�@     ��@     =�@     ��@     ��@     ��@     .�@     ��@     &�@     B�@     �@    ���@    �7�@     �@    ���@    ��@     ?�@     ��@    ���@    @ �@     ��@    �U�@     �@     ��@     �@     ��@     %�@     �@    �f�@    �O�@     �@     B�@     ��@     ݳ@     ��@     ̡@     <�@     �@      h@     �@@      @        
�
Conv2/h_conv2*�   `@     $#A! ��{Y�A)#�!;<�@2�	        �-���q=��|�~�>���]���>��~���>�XQ��>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�	            �
A              �?              �?              �?      �?              �?      �?      �?      �?              �?     �^@              �?      �?      �?      @       @      �?      @      @      �?      @      @      @      @       @      @      @       @      @      @      8@       @      "@      @      @      .@      $@      @       @      @      $@      (@      .@       @      .@      1@      6@      .@      5@      8@      6@      =@     �D@      D@     �P@      W@     �J@     �J@     �Q@      M@     �V@     @U@      X@     �U@     �Z@      i@     @_@     �n@     `g@     �c@     �v@     y@     0u@      o@     0r@     @r@     �v@     �@     �{@     �~@     ��@     0�@     ��@     ��@     X�@     �@     `�@     ��@     Ԓ@     ��@     ��@     `�@     ܛ@     h�@     �@     �@     n�@     ��@     R�@     X�@     ��@     =�@     ��@     ��@     ��@     .�@     ��@     &�@     B�@     �@    ���@    �7�@     �@    ���@    ��@     ?�@     ��@    ���@    @ �@     ��@    �U�@     �@     ��@     �@     ��@     %�@     �@    �f�@    �O�@     �@     B�@     ��@     ݳ@     ��@     ̡@     <�@     �@      h@     �@@      @        

cross_entropy_scl$�)A

training_accuracy)\>x~S�S      .��&	"7Z���Ad*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H  IDAT(�Ր1/Q�O6CF���((d����@�"�V�((%[��B�l�Q!l$
ɈD��D���e���y��m�y�˹�+���f7���I+�]q��* �����G�$����뒤�Z{*��8�����s����iz� ܅��x�P��Ӓ�性��6Pv%�%�'�*�ML��;<�����sò��c귃V�:���T$����Z�`�@9��E�c	�Z�J��Nŏ��� 'ɱ�^�v?
��*�%�y��ӒV��[�6������U	�7    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`` v�3.ɾ����g�&%��ֿ�����&y���"���߿��1���Ɣ<���!�?�0$/�����������yt9�+�VCX�_/��Iz���eN��a0as7�%U��)�ư���(I%V�L��C4#����
\畧|ά��0����(���?=0$������4�ҿ]��>��sb����������",r�!��݊�f� ��`�ӑߥ    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  &IDAT(���;KB��v+B� AH()$Q�^��A!}��!�5\[�)*�h�����hqp���"�T�W�SS��;Gg<�����Wq��I ��e[e3{�T��\/j�Y�/�� ��J~��H5���b���1 ׆F��?U��x�TN_����� 0� WN��i���FS��D�y��@L5�k}ݝ�ϓH�HJ��Vc�$�R�����jܒ�sy�˩�9�#ѹԨZ��Z�~�PnǴ�����Eg�������#z OZ�z-N���]��7~�c]k"r1he��.o��`�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@����	S����2�����`N(#A�C��B��2`J�>v���^b�Ldx2�����GSR؈a�v~���V������������L�J�0����p�X��$��� �lh���������sW*��.�0<>����������{��Ϭ````�����u4s�a��.
���.B��,�$�3�#��S�,�p�z�*>y*  ?�:�8S    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`6��-�jtn�џ�ԥ������?�X���|��"����+k9�CNy3000�������{Ȟ������o�d�)�?��Lp�ì#�9U����T��uϴw����x�<��,0Ɩ-�n����?6�d��1�� 0b�u��㔏8%���i%�SH``�)��A�<�Ȁ8��-�SR�ϟ��>�~���H  �!I�FH�    IEND�B`�
%
Conv1/weights/summaries/mean_1���
'
 Conv1/weights/summaries/stddev_1u��=
$
Conv1/weights/summaries/max_1i�G>
$
Conv1/weights/summaries/min_1�'N�
�
!Conv1/weights/summaries/histogram*�	   ���ɿ    M��?      �@!  ��� 	�)ҳ�Z~@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I����#@�d�\D�X=��.�?ji6�9�?�qU���I?IcD���L?ܗ�SsW?��bB�SY?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      &@      4@      .@      4@      ;@      0@      1@      5@      4@      3@      (@      0@      ,@      5@      2@      3@      ,@      *@      $@       @      @       @       @      @      @      @      @      @      @      @       @      @      @      �?      @       @      �?      �?       @      �?              �?      �?              �?              �?               @      �?              �?              �?              �?               @               @              �?              �?      �?       @      �?               @       @      �?      @      �?      @      @      @       @      @      @      @      @       @      "@      @      $@       @       @      @      $@      @      *@      *@      "@      1@      5@      (@      7@      0@      ,@      *@      5@      (@      0@      0@      1@      ,@      *@        
$
Conv1/biases/summaries/mean_1�k�=
&
Conv1/biases/summaries/stddev_1��;
#
Conv1/biases/summaries/max_1��=
#
Conv1/biases/summaries/min_1�m�=
�
 Conv1/biases/summaries/histogram*a	   ��m�?   �C8�?      @@!   �p-	@)����?28/�C�ַ?%g�cE9�?�������:              @@        
�"
Conv1/conv1_wx_b*�"	   �ӌ��   �F��?     $3A! ���@)w|�R���@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ뾢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾG&�$��5�"�g���0�6�/n��        �-���q=��n����>�u`P+d�>�����>
�/eq
�>��>M|K�>�_�T�l�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @      d@     P{@     ��@     ��@     ��@     d�@     ��@     Ĥ@     ئ@     :�@     ~�@     ��@     ��@     ��@     ��@     ��@     d�@     l�@     ߶@     ��@     ��@     H�@     ��@     Y�@     \�@     E�@     o�@     t�@     ��@     N�@     �@     ��@     N�@     ��@     Ԧ@     �@     ��@     0�@     �@     ��@     �@     ��@     4�@     <�@     ��@     ��@      �@     8�@     ؋@      �@     ȇ@     p�@     Ѓ@     ��@     (�@      ~@     0y@     �y@      v@     pu@     �r@     �q@     �o@     `l@      i@     �g@     �f@     �e@      c@     @_@     �\@     �[@     �Z@     �X@      V@      R@     �N@     �F@     �L@     �N@      F@      F@     �A@     �G@     �B@      6@     �C@      :@      ,@      6@      4@      1@      .@      .@      ,@      &@      &@       @      @      @      @      @      @      @       @      @      @      @      @      @      @      @       @       @      @      �?      �?              @              �?       @       @              �?      �?       @              �?               @      �?              �?              �?               @              �?              �?              �?      �?      �?               @      @      �?              @      @      @      @      @      �?      @      @      @       @      @      @      @       @      @      "@      0@       @      *@      ,@      .@      .@      .@      @      0@      3@      6@      ;@      ;@      7@      ?@      >@      C@      I@     �I@      J@      O@     �L@      P@      T@     @V@     �W@     �Y@     �W@     �Z@     @c@     �a@      e@     @f@     �g@     �g@     `n@     �r@     @q@      r@     �s@     Py@     px@     ~@     p~@     ��@     �@     @�@     ��@     P�@     �@     <�@     ��@     @�@     ��@     ��@     $�@     P�@     ؠ@     &�@     �@     Ĩ@     .�@     Ȯ@     �@     �@     -�@     ں@     E�@     ��@    �#�@     ��@    ���@    "�%A    ���@    �]�@    ��@    �Y�@    ��@    �V�@    ��@     V�@    ���@     ��@     ��@     ]�@     &�@     ��@    �w�@     ^�@     $�@     )�@     "�@     Z�@     �@     ��@     Ȇ@     �u@      `@      B@       @        
�
Conv1/h_conv1*�   �F��?     $3A!���2-1A))��^yr�@2�        �-���q=��n����>�u`P+d�>�����>
�/eq
�>��>M|K�>�_�T�l�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            �~A              �?               @              �?              �?              �?      �?      �?               @      @      �?              @      @      @      @      @      �?      @      @      @       @      @      @      @       @      @      "@      0@       @      *@      ,@      .@      .@      .@      @      0@      3@      6@      ;@      ;@      7@      ?@      >@      C@      I@     �I@      J@      O@     �L@      P@      T@     @V@     �W@     �Y@     �W@     �Z@     @c@     �a@      e@     @f@     �g@     �g@     `n@     �r@     @q@      r@     �s@     Py@     px@     ~@     p~@     ��@     �@     @�@     ��@     P�@     �@     <�@     ��@     @�@     ��@     ��@     $�@     P�@     ؠ@     &�@     �@     Ĩ@     .�@     Ȯ@     �@     �@     -�@     ں@     E�@     ��@    �#�@     ��@    ���@    "�%A    ���@    �]�@    ��@    �Y�@    ��@    �V�@    ��@     V�@    ���@     ��@     ��@     ]�@     &�@     ��@    �w�@     ^�@     $�@     )�@     "�@     Z�@     �@     ��@     Ȇ@     �u@      `@      B@       @        
%
Conv2/weights/summaries/mean_1��
'
 Conv2/weights/summaries/stddev_1V��=
$
Conv2/weights/summaries/max_1rTP>
$
Conv2/weights/summaries/min_1��P�
�
!Conv2/weights/summaries/histogram*�	    Tʿ   @�
�?      �@!��_@COL�)==ΐ3�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���>�?�s���O�ʗ���pz�w�7��})�l a��ߊ4F��iD*L�پ�_�T�l׾�_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              H@     ��@     0�@     (�@     ؏@     ��@     @�@     ��@     �@     �@     @�@     ��@     @�@     X�@     p�@     ��@     ��@     �@     @�@     ��@     H�@     ��@     �|@     �|@     �x@     @x@     �w@     @t@     Pq@     r@     �n@     @k@     �h@     �e@      d@     @b@     @`@      a@      Z@     �[@     �W@     �W@      Q@     @Q@     �M@      N@     �J@      E@     �H@      =@      8@      A@      A@     �B@      1@      6@      2@      9@       @      7@      (@      $@      0@       @      @      (@      *@      (@      @      &@      @      @      @      @      @       @       @      @      @       @      �?      @              �?      @      �?      �?       @              �?              �?              �?       @              �?              �?              �?              �?              �?               @       @      @               @      @      �?      @       @      �?       @      @              �?       @      @              @       @      @      "@      @      @      @      @      @      $@      $@      "@      .@      1@      0@      2@      8@      3@      6@      5@      =@      C@     �@@      >@     �A@     �@@      K@      G@      E@     �L@     �R@     �R@     �V@     �W@      X@      \@     �[@     �a@     �b@      a@     @e@      h@     @g@     �h@     0p@     pp@      r@      t@     �u@     Pw@     �x@     |@     0�@     p�@     ��@     ȃ@     ��@     H�@     ��@     `�@     X�@     h�@     T�@     |�@     $�@      �@     H�@     �@     ��@     H�@     ��@     P�@     x�@     x�@      *@        
$
Conv2/biases/summaries/mean_1��=
&
Conv2/biases/summaries/stddev_1`5 ;
#
Conv2/biases/summaries/max_1Q��=
#
Conv2/biases/summaries/min_1n_�=
�
 Conv2/biases/summaries/histogram*q	   ��K�?    ���?      P@!  ���0@)��dp���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:              �O@      �?        
�#
Conv2/conv2_wx_b*�#	    o5�   ����?     $#A! ���xd�@)p�|���@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾙ѩ�-߾E��a�Wܾ0�6�/n���u`P+d����n�����豪}0ڰ������R%������39W$:���:�AC)8g>ڿ�ɓ�i>��n����>�u`P+d�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              �?      @      9@     @S@     �h@     v@     P�@     �@     ��@     ��@     ��@     Ģ@     "�@     �@     ��@     �@     ��@     i�@    ��@     �@    ��@    ���@     �@     ��@     +�@     b�@    �0�@     ��@    �Q�@     ��@    �]�@    ���@    ��@     l�@     i�@     9�@     +�@     ��@     y�@     �@     ��@     ݲ@     ��@     �@     ��@     ��@     v�@     2�@     R�@     �@     ޡ@     ��@     h�@     ��@     Д@     ��@     ��@     ؏@     ��@     �@     ��@     ��@     H�@     Ѐ@     {@     �z@     P�@      x@     �q@     0x@     u@     �m@     �j@     0v@      d@     �y@     �c@      c@     �g@     �]@     @V@     �W@     �U@     @T@     @S@      U@      I@      I@     @P@     �E@     �V@     �S@      >@      4@      5@      6@     �@@      9@      3@      3@      "@      1@      2@      @      $@      *@      (@      @      @      @      @      @       @      @      @      @      @      @      �?      @      �?               @      �?      �?       @       @              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @      @               @               @              �?       @      �?       @      @       @      @       @      @      @      @       @       @       @      @      (@      @      @      @      *@      @      .@      3@      &@      *@      7@      3@     �k@      4@      >@     �E@     �C@     �P@      B@      C@      E@     �Q@      I@     @P@     @i@     �_@     �S@     �T@     �Z@     �Y@      g@     �b@     �g@     �g@      i@     �o@     �h@      o@     @t@     �p@     �y@     `}@     �y@     h�@      ~@     8�@     ��@     �@     ��@     ��@     \�@     ��@     �@     Ț@     |�@     l�@     D�@     Н@     ��@     �@     .�@     �@     (�@     ��@     &�@     ۰@     /�@     �@     �@     ��@     �@     ��@     ��@     ��@    ���@     o�@    �5�@     [�@     ��@    �p�@    ���@     :�@     ��@    ���@     c�@    ���@    �1�@    ���@    ���@    �Q�@     �@    ���@     ׼@     ��@     Ƴ@     "�@     �@     (�@     H�@     8�@      s@     �_@      D@      $@      �?        
�
Conv2/h_conv2*�   ����?     $#A! �yZ�\�@)��|�S��@2�        �-���q=:�AC)8g>ڿ�ɓ�i>��n����>�u`P+d�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�            ��A              �?              �?              �?              �?              �?               @      @               @               @              �?       @      �?       @      @       @      @       @      @      @      @       @       @       @      @      (@      @      @      @      *@      @      .@      3@      &@      *@      7@      3@     �k@      4@      >@     �E@     �C@     �P@      B@      C@      E@     �Q@      I@     @P@     @i@     �_@     �S@     �T@     �Z@     �Y@      g@     �b@     �g@     �g@      i@     �o@     �h@      o@     @t@     �p@     �y@     `}@     �y@     h�@      ~@     8�@     ��@     �@     ��@     ��@     \�@     ��@     �@     Ț@     |�@     l�@     D�@     Н@     ��@     �@     .�@     �@     (�@     ��@     &�@     ۰@     /�@     �@     �@     ��@     �@     ��@     ��@     ��@    ���@     o�@    �5�@     [�@     ��@    �p�@    ���@     :�@     ��@    ���@     c�@    ���@    �1�@    ���@    ���@    �Q�@     �@    ���@     ׼@     ��@     Ƴ@     "�@     �@     (�@     H�@     8�@      s@     �_@      D@      $@      �?        

cross_entropy_scl�K�?

training_accuracy)\?����U      ı��	�H\���A�*�
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�Ց�NA�?m��x�B��$&4�֘�p�l��x�B����
C�&6&�'� �@A�Y(�0��t4���|9��3���x�h��^��Ғsڒ
�š�����D���������[r���E�!5�n\ږ���N��y��L���Q6��Б���L n� �0\ wG���+?��(R��xƠ���<z�ҁ�J�IÞ$mj��+���%I���Zv�T��    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�ő��0�Bt&*��l9�X�_A7{�>3��<؂���A����个��W�L��]��m�#&r� ��uצw�j���p��m�ǀ�����2��yBz�_]��	C��Z ����S�{�Z���r>9n�5�EQ    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  IDAT(���=HBQ��'	BQqH8�]�jJ�ph�[
� [\��5��!j��U�����Z*P0�c4�����=�h��^~�y��9ҿ�������	�5 h5���.@�>"I
�8�q�/��W��� �4r����1�~cL̤�;��{�z�lfEϋ� LN�$M�G̼pU��fݏ�(Ǖ���Cu(ĥ�9�	Ks�Ø$�B�B��Y^��&��	�����~w�#Dz$9�9���㋼��q�l�?��Ru����:�/w�t��I�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H  IDAT(��ұJA��FD�i{-T�M@!����		Z��|� �E ����`ek�F�c��ue�*�?h��8;iou��sa������>M=4��wm#� �sQ��~c%����,�T$�-yޕ�Ի����;lg�s�ԭ�Pp��|�J�8�pڲ���J�:pgk@#MqͲ�W���C;9�}�?)��%�p^���>�uN�Joi���>\JR1 �v���4���q#hJ^�"(�׺`zլ)���I)KR�<c�����b��� �t �;>�_�2)���#�    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@���pLQ&U��+KQ�����^b```P|�7C��/�_3�"$�!�;�c�300��~��n�쿏d.��)�f,�;�
L�J�I���''�̰����{�������_�>����߿����{��_	tc�	�d��C
��d,�43Ia<�@/�c�T�A��  a�<u?;;    IEND�B`�
%
Conv1/weights/summaries/mean_1�"��
'
 Conv1/weights/summaries/stddev_1�~�=
$
Conv1/weights/summaries/max_1��H>
$
Conv1/weights/summaries/min_1��N�
�
!Conv1/weights/summaries/histogram*�	   ���ɿ   �7�?      �@!  ���	�)lS[Cu@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��m9�H�[���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�+A�F�&�U�4@@�$�})�l a�>pz�w�7�>�!�A?�T���C?<DKc��T?ܗ�SsW?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      &@      4@      ,@      5@      ;@      *@      5@      5@      3@      2@      *@      1@      ,@      3@      2@      3@      0@      &@      $@      @      &@      @       @      @       @       @      @      @      @      @      �?       @      @       @       @       @      �?       @      �?      �?       @              �?      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?              �?              @      @               @      �?      @      @      @      @      @      @      @      @       @      &@      @      $@      @      $@      @      "@      *@      *@      3@      .@      .@      7@      0@      *@      ,@      4@      &@      3@      0@      0@      ,@      *@        
$
Conv1/biases/summaries/mean_1�=
&
Conv1/biases/summaries/stddev_1��#;
#
Conv1/biases/summaries/max_1̭�=
#
Conv1/biases/summaries/min_1���=
�
 Conv1/biases/summaries/histogram*q	   `�<�?   ��U�?      @@!   a#	@)d^���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               >@       @        
�"
Conv1/conv1_wx_b*�"	   �"T�   ���?     $3A! \�����@)��P��P�@2��P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾��~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž�XQ�þ��~��¾G&�$��5�"�g����H5�8�t>�i����v>��n����>�u`P+d�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�             @\@     �y@     ��@     ��@     l�@     ^�@     L�@     �@     ��@     (�@     �@     ��@     �@     �@     ��@     u�@     ��@     `�@     ��@     4�@     ��@     {�@     ,�@     8�@     !�@     ڳ@     �@     h�@     �@     ]�@     $�@     ��@     D�@     J�@     ��@     Ҥ@     �@     �@     ��@     (�@     �@     ��@     ��@     Е@     $�@     ��@     ��@     �@     ��@     0�@     �@     H�@     ��@     ��@     @@     �~@     �z@     �v@     v@     �t@     �p@     �p@     �n@     �m@      j@     �f@     �d@     �d@     �`@     @\@     �\@     @V@     �V@      X@     �U@     �Q@     �K@      O@     �I@     �F@      B@     �B@     �G@     �C@      3@      =@      6@      5@      1@      1@      4@      0@      (@      5@      $@      $@      *@      $@      $@      @       @      "@      @      @      @      @      @      @      @      @      @       @       @              @              �?               @      �?              �?              �?              �?               @              �?              �?              �?              �?      �?      �?      �?      �?      �?      �?      �?               @       @               @       @              @      �?               @      @      �?      �?       @      @      @       @       @      �?      @      @      �?      @      @      @       @      @      @       @      @      @      @      @      *@      .@      0@      .@      7@      2@      ?@      7@      B@      B@      ;@     �C@      >@     �D@     �J@     �E@     �H@     �O@     @Q@     �Q@     @W@      Y@     �V@      X@     @]@      Z@     �b@      d@     �e@      j@      k@      l@      p@     �p@     �s@     pu@     �v@      z@      @     �~@      �@     ��@     p�@     ��@     p�@     `�@     $�@     �@     ��@     <�@     x�@     ��@     �@     
�@     ��@     ��@     l�@     <�@     0�@     �@     9�@     ��@     ��@     �@    ���@     .�@    ���@     5�@    �V$A    @��@    ���@    ���@    ��@     
�@    ��@     ��@     �@    �L�@    �O�@    �(�@     ��@    �8�@     ��@     ��@     Y�@     ��@     A�@     �@     Z�@     ��@     l�@     @�@     �w@      _@      ;@      @        
�
Conv1/h_conv1*�   ���?     $3A! ��h~?A)_�@���@2�	        �-���q=�H5�8�t>�i����v>��n����>�u`P+d�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�	            ��A              �?              �?              �?      �?      �?      �?      �?      �?      �?      �?               @       @               @       @              @      �?               @      @      �?      �?       @      @      @       @       @      �?      @      @      �?      @      @      @       @      @      @       @      @      @      @      @      *@      .@      0@      .@      7@      2@      ?@      7@      B@      B@      ;@     �C@      >@     �D@     �J@     �E@     �H@     �O@     @Q@     �Q@     @W@      Y@     �V@      X@     @]@      Z@     �b@      d@     �e@      j@      k@      l@      p@     �p@     �s@     pu@     �v@      z@      @     �~@      �@     ��@     p�@     ��@     p�@     `�@     $�@     �@     ��@     <�@     x�@     ��@     �@     
�@     ��@     ��@     l�@     <�@     0�@     �@     9�@     ��@     ��@     �@    ���@     .�@    ���@     5�@    �V$A    @��@    ���@    ���@    ��@     
�@    ��@     ��@     �@    �L�@    �O�@    �(�@     ��@    �8�@     ��@     ��@     Y�@     ��@     A�@     �@     Z�@     ��@     l�@     @�@     �w@      _@      ;@      @        
%
Conv2/weights/summaries/mean_1eW��
'
 Conv2/weights/summaries/stddev_1�=
$
Conv2/weights/summaries/max_1�RP>
$
Conv2/weights/summaries/min_1r�R�
�
!Conv2/weights/summaries/histogram*�	   @Pʿ   �[
�?      �@! ($U�L�)H"�:�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����Zr[v��I��P=��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�;�"�qʾ
�/eq
Ⱦ�u��gr��R%������39W$:��>R%�����>['�?��>K+�E���>jqs&\��>��~]�[�>8K�ߝ�>�h���`�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              G@     ��@     `�@     h�@     �@     X�@     H�@     l�@     <�@     ��@     ��@     �@     @�@     0�@     8�@     ��@     p�@     ��@     P�@     �@     ��@      �@     `}@     �}@     �v@     py@     �w@     �t@     �q@     �q@     @n@     �n@      h@      d@     �d@     @^@     ``@     �`@     �^@      Y@      W@     @W@     @R@      P@     �M@      M@     �L@     �F@     �G@      F@      @@      @@      =@      :@      8@      9@      7@      0@      .@      ,@      *@      $@      "@      (@      3@      $@      @       @      &@      @      @      @       @      @      @      @      @      @      @      �?      @       @      �?      @       @       @      �?      �?       @      �?       @       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      @      �?               @      �?              @      �?      �?       @      @      @      �?      @      @      @      @      @      @      @      &@      (@       @       @      *@      *@      .@      6@      *@      4@      2@      6@      <@      B@     �E@     �B@     �E@      L@     �A@     �B@      J@      N@     �K@     �Q@     �W@      S@     �^@     �[@      ^@     �a@     �`@     �a@     `d@     `h@     `j@     @h@      o@      p@      r@     �t@     �t@     �x@     0x@     �{@     8�@     �@     Ђ@     �@     @�@     ��@     ��@     �@     �@     0�@     0�@     |�@     L�@     ��@     X�@     �@     ؒ@     <�@     @�@     �@     ��@     H�@      .@        
$
Conv2/biases/summaries/mean_1�s�=
&
Conv2/biases/summaries/stddev_1��;
#
Conv2/biases/summaries/max_1X�=
#
Conv2/biases/summaries/min_1�d�=
�
 Conv2/biases/summaries/histogram*q	    ��?   ��?      P@!  ��}.@)�?t<��?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:              �N@      @        
�%
Conv2/conv2_wx_b*�%	    ��   ��a�?     $#A! `9ź�@)�у�(�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ5�"�g���0�6�/n���u`P+d����n������5�L�����]�����MZ��K���u��gr���5�L�>;9��R�>0�6�/n�>5�"�g��>G&�$�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              �?      @     �@@     �W@     �h@     �u@     Ȃ@     @�@     ܑ@     ؗ@     ��@     �@     ��@     �@     �@     ��@     �@     ��@    �u�@    ���@    ���@     l�@    ���@     ��@     ��@    ���@    ���@     �@    ���@     0�@     _�@     .�@     п@     �@     @�@     -�@     �@     �@     ��@     W�@      �@     �@     ��@     ~�@     ��@     ��@     Ԧ@     �@     �@     ��@     ȟ@     ��@     h�@     �@     $�@     �@     @�@     |�@     �@     ��@     ��@     H�@     0�@     �@     �}@     @�@      x@     �x@     �r@     �s@      n@     }@     �n@     �d@     �d@     �`@      `@     �[@     `d@     �^@     @S@     �T@     �_@     �S@     �V@     @R@      P@      F@      ?@      E@     �Q@     �A@      A@     �D@      >@      <@      :@      *@      0@      &@      6@      "@      (@      &@      &@      @      �?      @      @      @      @      @      @      @      @               @      �?      �?               @      @      �?      �?      �?      �?               @              �?       @      �?      �?      �?       @               @               @      �?      �?               @              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?               @              �?      �?      �?               @              7@      �?       @      �?              @      �?              @      @      D@       @      @      @      @      (@       @      @      @      $@       @       @      &@      .@      0@      2@      3@      2@      9@      >@      2@      C@      ?@      L@      F@     �@@     �E@     �H@     �R@      U@     �L@      R@      \@     �W@      W@     �]@     �_@     `a@     @b@     �m@      f@     @i@     `k@     �q@     0p@     �r@     `u@     �|@     �z@     �z@     Py@     �x@     І@     ��@     ؋@     �@     ��@     P�@     L�@     Ԑ@     T�@     �@     ��@     ��@     x�@     ��@     ޥ@     N�@     @�@     (�@     N�@     0�@     Ԯ@     6�@     )�@     B�@     ��@     ��@     ]�@     �@     �@     }�@     *�@     �@     ��@    ���@     ?�@    ���@     0�@     ��@    �;�@    ��@     ��@     ��@    ���@    ���@    ���@     ��@     {�@     �@     >�@     ٱ@     ��@     �@     8�@     �@     `{@     �m@     �S@      5@      @      �?        
�
Conv2/h_conv2*�   ��a�?     $#A! -A�#u�@)�nT�]�@2�	        �-���q=�5�L�>;9��R�>0�6�/n�>5�"�g��>G&�$�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�	            �A              �?              �?      �?              �?      �?      �?              �?               @              �?      �?      �?               @              7@      �?       @      �?              @      �?              @      @      D@       @      @      @      @      (@       @      @      @      $@       @       @      &@      .@      0@      2@      3@      2@      9@      >@      2@      C@      ?@      L@      F@     �@@     �E@     �H@     �R@      U@     �L@      R@      \@     �W@      W@     �]@     �_@     `a@     @b@     �m@      f@     @i@     `k@     �q@     0p@     �r@     `u@     �|@     �z@     �z@     Py@     �x@     І@     ��@     ؋@     �@     ��@     P�@     L�@     Ԑ@     T�@     �@     ��@     ��@     x�@     ��@     ޥ@     N�@     @�@     (�@     N�@     0�@     Ԯ@     6�@     )�@     B�@     ��@     ��@     ]�@     �@     �@     }�@     *�@     �@     ��@    ���@     ?�@    ���@     0�@     ��@    �;�@    ��@     ��@     ��@    ���@    ���@    ���@     ��@     {�@     �@     >�@     ٱ@     ��@     �@     8�@     �@     `{@     �m@     �S@      5@      @      �?        

cross_entropy_scl��?

training_accuracy��L?z�iPT      |��2	&to^���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`� �7G[%�τ$�R��Ce� 6��J28~���ԫk���	@26B��A9���$k"�r����j/�:a����YBb�߿X;�/d�E������cq������+c����˄"�����).3,�{���؍�I85������E�Ӑ��S���?�\ɀG�G��I� Vn%  }�=V=��    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`��5����flR�i��}���6���NID-�*�r�g!C�S,�m�Q����tF�F����SŔ��/��A)��Q^��T^�N[a�G���t�߿��������I��-��*���`��!.9���p�������W��//,�h�)'|�N�x4
]8ςSR�?n�X  �,A�Ѵ{    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c`��)��rLQ%6��}��bJ�?-X��߿w<0������߿�>�p����Ϯ�x�3wp�xl!��8���1��?�����߿������E�{�������O\�	��z���u��L1�̅�����N�	���)��|�NI���Bt���)�z�a}.��/������|5o�ÿ���1�8�����߿gSX�hd���߿�8CE  �^Y��bB�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`pP���Ǆ*���nIW|��%M��T���N������m��\�q;(��NIY'F�ne���O�N]�3p����IT�"����B�'�qJ��{��䗿Sp��&��	  31F]4��    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`d�Js��10��~��#��Eg�����￿��#�Y£�:�`-��4Et��z�Z�G���t��ܒ̫�}�%����?.9�g���j���:/V)��o�����D�=ɯ��=������	cH�H2000H��e6���3�,&L�k�85���� ������;�K�V>��"����Á��`7s�۟��Β�n�ο��p:�z  �Tm|)|�    IEND�B`�
%
Conv1/weights/summaries/mean_1Ĕx�
'
 Conv1/weights/summaries/stddev_1�x�=
$
Conv1/weights/summaries/max_1I>
$
Conv1/weights/summaries/min_1��N�
�
!Conv1/weights/summaries/histogram*�	    ��ɿ   @""�?      �@!  F�F�)K����r@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�nK���LQ�k�1^�sO�IcD���L���82���bȬ�0�x?�x��>h�'���lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      &@      2@      1@      3@      9@      0@      5@      6@      3@      1@      *@      1@      ,@      4@      .@      5@      0@      &@      "@      $@      "@      @       @      @      @      @       @      �?      @      @       @       @       @       @      @      @      �?      �?      �?       @      �?      �?      �?              �?              �?              �?               @      �?              �?              �?              �?              �?      �?      �?              �?       @      �?              �?       @      �?      @      @       @      �?      �?      @      @      @      @       @      @       @      @      @      ,@      @      (@      @      $@      @       @      ,@      (@      1@      0@      1@      4@      3@      *@      .@      3@      "@      5@      0@      0@      0@      &@        
$
Conv1/biases/summaries/mean_1���=
&
Conv1/biases/summaries/stddev_1˳-;
#
Conv1/biases/summaries/max_12�=
#
Conv1/biases/summaries/min_1�T�=
�
 Conv1/biases/summaries/histogram*q	   `�*�?   @���?      @@!   �4	@)y8����?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               >@       @        
�#
Conv1/conv1_wx_b*�"	    ���    �&�?     $3A! }UF<}�@)$�ѽC�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE�����_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ��z!�?�>��ӤP��>.��fc��>39W$:��>��|�~�>���]���>�u`P+d�>0�6�/n�>��~���>�XQ��>�����>
�/eq
�>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�               @     �R@     �t@     x�@     ��@     ��@     `�@     0�@     2�@     ��@     ڧ@     *�@     P�@     ��@     �@     ��@     ��@     c�@     ��@     Ե@     ��@     �@     ��@     ��@     �@     &�@     ��@     ��@     �@     A�@     ��@      �@     �@     ب@      �@     N�@     ��@     ��@     ��@     ��@     ��@     �@     �@     �@     ��@     ܓ@     ��@     �@     Ў@     �@     ��@     (�@     ��@     �@      �@     `~@      |@     �z@     @w@      v@     �t@     pr@     p@     @p@     �k@     �i@      g@     �e@     �c@      c@     ``@     @]@     �Z@      Z@      W@     @W@     �P@     �R@      J@      K@     �E@     �E@     �B@      :@      ?@      :@      >@      3@      5@      7@      7@      5@      *@      "@      1@      &@      &@      "@      "@      0@      $@      @      @      @      @      @       @      @      �?       @       @      @      @      @      @      @       @      �?      �?       @       @       @              �?      @              �?      �?       @      �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?      �?      �?              �?      �?              �?      �?       @       @      �?      @       @      @      @      @       @      �?       @      @       @      @      @      @      @      "@      @      $@      (@      0@      "@      (@      1@      .@      0@      *@      4@      1@      ;@      >@     �@@      >@     �@@      E@     �G@     �H@      I@      K@     �N@     �T@     �V@     �U@     �T@     �Y@     �]@     �^@     �a@     �b@     �d@     �d@     `k@     `l@     `n@     0q@     Pt@     �r@     �u@     �y@     }@     P~@     �@     0�@      �@     `�@     0�@     ��@     ��@     Ȑ@     �@     ��@     ؖ@     ��@     ��@     <�@     ��@     ��@     ��@     ��@     Э@     ��@     R�@     |�@     ��@     �@    ���@     [�@    �)�@    ���@    �%A    �@     Y�@    �M�@    ���@    ���@     '�@     ��@    ��@    ��@    �|�@    �"�@    ���@     Z�@     ��@    ���@     ��@     S�@     �@     ��@     ��@     ��@     ��@     �@     `q@     �W@      *@      �?        
�
Conv1/h_conv1*�    �&�?     $3A!���S�A)D2�v �@2�	        �-���q=��z!�?�>��ӤP��>.��fc��>39W$:��>��|�~�>���]���>�u`P+d�>0�6�/n�>��~���>�XQ��>�����>
�/eq
�>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�	            XAA              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?      �?      �?              �?      �?              �?      �?       @       @      �?      @       @      @      @      @       @      �?       @      @       @      @      @      @      @      "@      @      $@      (@      0@      "@      (@      1@      .@      0@      *@      4@      1@      ;@      >@     �@@      >@     �@@      E@     �G@     �H@      I@      K@     �N@     �T@     �V@     �U@     �T@     �Y@     �]@     �^@     �a@     �b@     �d@     �d@     `k@     `l@     `n@     0q@     Pt@     �r@     �u@     �y@     }@     P~@     �@     0�@      �@     `�@     0�@     ��@     ��@     Ȑ@     �@     ��@     ؖ@     ��@     ��@     <�@     ��@     ��@     ��@     ��@     Э@     ��@     R�@     |�@     ��@     �@    ���@     [�@    �)�@    ���@    �%A    �@     Y�@    �M�@    ���@    ���@     '�@     ��@    ��@    ��@    �|�@    �"�@    ���@     Z�@     ��@    ���@     ��@     S�@     �@     ��@     ��@     ��@     ��@     �@     `q@     �W@      *@      �?        
%
Conv2/weights/summaries/mean_1�]��
'
 Conv2/weights/summaries/stddev_1��=
$
Conv2/weights/summaries/max_1��P>
$
Conv2/weights/summaries/min_1g�S�
�
!Conv2/weights/summaries/histogram*�	   �pʿ   `��?      �@!�W��W�L�)�9DC3�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]�����[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���uE���⾮��%ᾞ[�=�k���*��ڽ�����>豪}0ڰ>�����>
�/eq
�>K+�E���>jqs&\��>���%�>�uE����>�f����>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �G@     x�@     0�@     x�@     �@     T�@     $�@     p�@     `�@     ��@     d�@     �@     ��@     ��@     ��@     `�@     ��@     Ȉ@     �@     ��@     ��@     (�@     `~@     p|@     @w@     0y@     `x@      t@     �q@     Pq@      o@     `l@     �g@      g@     �b@     �`@     @`@     @]@     @]@      ]@      U@     @X@      T@     @R@     �L@     �G@     �L@      H@      D@     �E@      J@      >@      :@      :@      ?@      2@      5@      .@      &@      ,@      5@      @      &@      ,@      "@      (@      @      "@      @      @      @      @      @      @      �?      �?       @       @      @      @       @              @      @      �?              @               @              �?      �?      �?              �?              �?              �?              �?              �?              �?               @      �?              �?      @      �?              �?              @      �?       @      �?              �?      �?              �?      �?      �?      @      @      @      �?      @      �?      @      @      @       @      @      @       @      "@      &@       @      (@      0@      0@      2@      4@      ,@      6@      6@      ;@     �A@     �B@      A@     �A@     �G@     �G@      I@     �I@     �K@      N@     �U@      V@      T@     �Y@     �]@     �[@      ]@     ``@     �e@      e@     �e@     �h@     �k@     @o@     �p@     �p@     �u@     Pt@     x@     �x@     �}@     @@     ��@     x�@     ȃ@     p�@     �@     ��@     x�@     @�@     Ȏ@     �@     ��@     @�@     ��@      �@     �@     Ԓ@     <�@     h�@     ��@     ��@     `�@      4@        
$
Conv2/biases/summaries/mean_1̀�=
&
Conv2/biases/summaries/stddev_1�H);
#
Conv2/biases/summaries/max_1��=
#
Conv2/biases/summaries/min_1&w�=
�
 Conv2/biases/summaries/histogram*q	   ���?   @��?      P@!  ��0@)|������?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               N@      @        
�#
Conv2/conv2_wx_b*�#	    U��   @(��?     $#A! �H`��@)/wv�iH�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ����ž�XQ�þ��~��¾�[�=�k���4[_>������m!#��ہkVl�p>BvŐ�r>�u`P+d�>0�6�/n�>�����>
�/eq
�>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?�������:�              @      9@     �Q@     `e@      t@     ��@     ��@     ��@     L�@     ĝ@     t�@     Z�@     Q�@     ��@     ��@     T�@     "�@    � �@     .�@    ���@     A�@     �@    ���@     �@    �#�@     H�@    ���@    ���@    �_�@     ��@    ��@     v�@     ս@     .�@     ۺ@     ��@     η@     ��@     ��@     S�@     ��@     �@     ƨ@     ��@     0�@     �@     $�@     ��@     x�@     �@     Н@     r�@     �@     ��@     P�@     @�@     �@     ��@     Ј@     ��@     0�@     �@      �@     `@     `�@     u@     �s@     0�@      u@     �q@     �j@     �k@     �p@     �g@      g@     `c@     @^@     �^@     @_@      Z@     @V@      Z@     �S@     @d@      V@      L@      G@     �@@      p@      K@      C@      6@      9@      8@      3@      :@      .@      2@      (@      $@      $@      2@      "@      &@      *@      @       @      "@      @      @      @      @      @      @      @       @      @       @      @       @      �?              �?              �?      @      �?       @              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      @      �?              �?              �?      �?      @              �?       @       @       @      @      @       @       @       @      �?       @      @      �?      @      @      @      @      @      @      $@      @       @      @      (@      @      .@       @      6@     �n@      1@      0@      :@      <@     �A@      =@      :@      A@     �@@     �M@     �F@     @W@      O@     �P@     �c@      Q@     @T@     @a@     �Z@     @c@      ]@     `a@     �p@     @e@     �h@     �l@     pq@     �p@     �o@     0s@     �@     �@     ��@     Ђ@     �{@     �@     �@     ��@     ��@     (�@     x�@     �@     4�@     ��@     h�@     h�@     ��@      �@      �@     ��@     ��@     ޥ@     $�@     �@     `�@     ��@     ��@     ��@     H�@     �@     ɼ@     ��@     ��@     r�@     9�@     ��@    �.�@    ���@    ��@     ��@     .�@    ���@    ���@     ��@     /�@     �@    ���@     .�@     ��@    ���@    �L�@     �@     �@     ?�@     z�@     �@     d�@     ��@      �@      q@     �a@     �B@      *@        
�
Conv2/h_conv2*�   @(��?     $#A! UX��@)��0��@2�        �-���q=ہkVl�p>BvŐ�r>�u`P+d�>0�6�/n�>�����>
�/eq
�>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?�������:�            D�A              �?              �?              �?              �?      @      �?              �?              �?      �?      @              �?       @       @       @      @      @       @       @       @      �?       @      @      �?      @      @      @      @      @      @      $@      @       @      @      (@      @      .@       @      6@     �n@      1@      0@      :@      <@     �A@      =@      :@      A@     �@@     �M@     �F@     @W@      O@     �P@     �c@      Q@     @T@     @a@     �Z@     @c@      ]@     `a@     �p@     @e@     �h@     �l@     pq@     �p@     �o@     0s@     �@     �@     ��@     Ђ@     �{@     �@     �@     ��@     ��@     (�@     x�@     �@     4�@     ��@     h�@     h�@     ��@      �@      �@     ��@     ��@     ޥ@     $�@     �@     `�@     ��@     ��@     ��@     H�@     �@     ɼ@     ��@     ��@     r�@     9�@     ��@    �.�@    ���@    ��@     ��@     .�@    ���@    ���@     ��@     /�@     �@    ���@     .�@     ��@    ���@    �L�@     �@     �@     ?�@     z�@     �@     d�@     ��@      �@      q@     �a@     �B@      *@        

cross_entropy_scl��>

training_accuracy�Ga?5ҳ�S      ���%	��`���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H  IDAT(���;K�P��*xI?�'I%������V28ttg7Al��
��_@q��CA�������z�z��}���߹���3����\i;N�������4; @x�$,wO�jGF/���1z2����J�v�h׳6�}�@P����*���F�I���h�%���b_�4���i �l�}�R~/�ăv _�|�/Ϻ�GRc��|�v:M��u� X�=�6nmԽq��#j����G	��\hmϦ����"���<�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H  IDAT(�c`�?`Fb��j�:���g}�]�[꽿�������}P-,09��J���^y�ݟ��'��IZ~�������a7�W\����w'+9��Ub�����;�rj����;��˖7����@�䖿�0&�Bo=��Vv�U�h�kL����dƔ��g�C=ʄMZ�.ɬ���G(܎_�2�y�ߒR����+�00|e��x��a���9n��WŌ������ۧ�����w�8�PFF���ޝ�^���>� ��j���L�    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  IDAT(���?(�q��w�"��%�,�܆D)�JJ���F����N�ҍl�+gP���2�t�	E�P�_�_o�o�~��gz����<=��O(y^܌Rd��-��v�3l��3<8N��*��w!&%ر� �Ij<bдS^*_}���j���Z��W��y�1C1.�fS�m�K���W)ʭn�_-I����6l�s'%I���a5f��I�'�����<�zK,KR�%�:�T�W�,���iY�K8�q��b͔F`���$|k�A&P�����/,.r�}��I    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H  IDAT(�͑;KQ�	����V"�b'X�H�&��v>Za�B��؉Uz,|� !6V�Q,b����$>��"�/;�9�������3MI�hdS6�
��T�`���ށ��2���W��Ijs`Gk����ty�� �������NR�.��޺쓶.��+�)����9�>h�Sْ��v�
V��I�3�Kp��!oOG���.INԸ����2�*T=OSu����O؏�5����]^��y^�In$x��Z�x4�)���0����@~��r�    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   nIDAT(�c`h`w�*2�ERC�?nIF���(:Q��q�d@Չ*y�����亖a t!QQ:���ϰH����_�����=�p����#�.Z�gn��  �B(T�~;    IEND�B`�
%
Conv1/weights/summaries/mean_1�r�
'
 Conv1/weights/summaries/stddev_1���=
$
Conv1/weights/summaries/max_1�\J>
$
Conv1/weights/summaries/min_1`jO�
�
!Conv1/weights/summaries/histogram*�	    L�ɿ    �K�?      �@! �.,���)G]�p�w@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��qU���I�
����G��T���C��!�A�6�]���1��a˲�O�ʗ��>>�?�s��>�qU���I?IcD���L?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      (@      1@      0@      3@      <@      .@      4@      5@      2@      3@      *@      1@      *@      3@      1@      4@      0@      $@      *@      @      $@      @      @      @      @       @       @      @      @      @       @              @       @      �?      @       @               @       @      �?              �?               @               @              �?              �?              �?              �?              �?      �?      �?              �?              �?      �?              �?      �?              �?      �?      @              @      �?      @              �?      @       @      @      �?      @      @      @      @      (@      @      &@       @       @      @      "@      ,@      *@      .@      0@      3@      5@      2@      (@      (@      5@      "@      7@      ,@      .@      1@      (@        
$
Conv1/biases/summaries/mean_1l��=
&
Conv1/biases/summaries/stddev_1�6;
#
Conv1/biases/summaries/max_1���=
#
Conv1/biases/summaries/min_1�}�=
�
 Conv1/biases/summaries/histogram*q	   ���?   ��z�?      @@!   om	@)'<P�I��?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               =@      @        
�"
Conv1/conv1_wx_b*�!	   @���   ��?     $3A! �n�d�@)����@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾
�/eq
Ⱦ����ž�XQ�þ��~��¾���]���>�5�L�>G&�$�>�*��ڽ>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @     �b@     �@     (�@     ��@     ��@     Л@     >�@     ��@     r�@     Ʃ@     �@     Z�@      �@     	�@     m�@     V�@     �@     ��@     p�@     ��@     ��@     �@     ��@     �@     �@     Ҳ@     ��@     B�@     �@     J�@     �@     �@     \�@     �@     ��@     ��@     
�@     ��@     T�@     l�@     ��@     Ș@     ��@     $�@     ��@     ��@      �@     �@     H�@     ��@     p�@     ��@     H�@     H�@     `}@     �|@     `z@     �x@     �t@     0t@      r@     �n@     @k@     �j@     �f@     �e@     �d@     `a@     �^@      \@     ``@     @^@      X@     �T@      S@     �O@      R@     �Q@     �N@      F@      H@     �B@      E@      C@      =@      =@      2@      :@      3@      7@      .@      .@      ;@      *@      &@      "@      *@      &@      (@      (@      @      @      @      @      @       @      @      �?       @       @      �?      �?       @      �?               @       @      @      �?      �?       @      @              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              �?       @      @      �?      @       @       @      �?      @      @      �?       @      @      @      �?      @      @      @      @      @      @      @      @      @      *@      &@      @       @      $@      1@      3@      ,@      4@      1@      2@      9@      ,@      B@      1@      G@      I@     �B@      D@     �H@      N@      H@     �O@      M@     @V@     �T@      U@      X@     @[@      b@      b@     �_@     @d@     �e@      i@     �j@     �p@      r@     Ps@     �t@     �v@     �x@     �{@     �~@     0�@     @�@     ��@     ��@     ��@     `�@     <�@     ,�@     ��@     ȕ@     ��@     ��@     ��@     ��@     ��@     L�@     ��@     ̩@     v�@     װ@     �@     C�@     ��@     �@     ��@     ��@    �a�@    ���@    o$A    �8�@     ��@     &�@     (�@    ���@     �@    �A�@    �\�@     R�@     /�@    �u�@     ��@    ���@     4�@     �@     ��@     @�@     �@     .�@     f�@     l�@     ��@     X�@     �s@     �Y@      1@      @        
�
Conv1/h_conv1*�   ��?     $3A! �JoA)�3,	��@2�        �-���q=���]���>�5�L�>G&�$�>�*��ڽ>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            (�A              �?              �?              �?              �?      �?              �?       @      @      �?      @       @       @      �?      @      @      �?       @      @      @      �?      @      @      @      @      @      @      @      @      @      *@      &@      @       @      $@      1@      3@      ,@      4@      1@      2@      9@      ,@      B@      1@      G@      I@     �B@      D@     �H@      N@      H@     �O@      M@     @V@     �T@      U@      X@     @[@      b@      b@     �_@     @d@     �e@      i@     �j@     �p@      r@     Ps@     �t@     �v@     �x@     �{@     �~@     0�@     @�@     ��@     ��@     ��@     `�@     <�@     ,�@     ��@     ȕ@     ��@     ��@     ��@     ��@     ��@     L�@     ��@     ̩@     v�@     װ@     �@     C�@     ��@     �@     ��@     ��@    �a�@    ���@    o$A    �8�@     ��@     &�@     (�@    ���@     �@    �A�@    �\�@     R�@     /�@    �u�@     ��@    ���@     4�@     �@     ��@     @�@     �@     .�@     f�@     l�@     ��@     X�@     �s@     �Y@      1@      @        
%
Conv2/weights/summaries/mean_1���
'
 Conv2/weights/summaries/stddev_1:�=
$
Conv2/weights/summaries/max_1XsQ>
$
Conv2/weights/summaries/min_1OT�
�
!Conv2/weights/summaries/histogram*�	   �ɀʿ    k.�?      �@!@�o@bJ�)f��.}�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�K+�E��Ͼ['�?�;��n�����豪}0ڰ�;9��R���5�L����(���>a�Ϭ(�>8K�ߝ�>�h���`�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �H@     H�@     H�@     p�@     ��@     ��@     ��@     ��@     @�@     Г@     $�@     4�@     |�@     �@      �@     x�@     ��@     ��@     ��@     ��@      �@     ��@      @     0|@     0w@     �x@     �x@     �s@     @r@     Pq@      o@      m@     `h@     `e@      c@     �b@     @\@     �^@     �_@      X@     �[@      T@     �T@     �Q@     @Q@     �G@     �N@      C@     �E@      =@      ?@      E@      ?@      =@      5@      5@      1@      6@      5@      3@      1@      "@      4@      (@       @      "@      "@      @      @      @      @      @      @      @      @      @       @      @      @      @       @       @       @      @      �?      �?      �?      �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?      �?       @      @      @      �?      @       @       @      �?      �?               @      @      @      @      @      @      @      @       @      @      *@      .@      &@      "@      .@      ,@      "@      9@      1@      8@      >@      @@      >@      C@     �C@     �B@      ?@     �H@      H@      I@     �O@     �N@      Q@      V@     �V@     �Z@     �[@      \@     @_@      `@      c@      g@     `f@     �i@     �k@     �p@     p@     �p@     Pt@      u@     �v@     �z@      ~@     0@     �@     ��@     x�@     @�@     ��@     �@     P�@     ��@     ��@     Ȑ@     ؒ@     \�@     ��@     ,�@     D�@     Ē@     @�@     p�@     ��@     (�@     X�@      6@        
$
Conv2/biases/summaries/mean_1Ƶ�=
&
Conv2/biases/summaries/stddev_1�4;
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1���=
�
 Conv2/biases/summaries/histogram*q	   @��?   ���?      P@!   Ǹ6@)��ϴ���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               M@      @        
�"
Conv2/conv2_wx_b*�"	   ��% �    O��?     $#A!  �Ӆ�@)�8����@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾K+�E��Ͼ['�?�;0�6�/n���u`P+d���5�L�>;9��R�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              @      9@     �S@      g@      x@     x�@     H�@     ��@     ��@     �@     R�@     ��@     u�@     �@     �@     ޽@     ��@     d�@     ��@     ��@    �E�@    �$�@    �^�@    ���@    ���@    ���@     ��@    ���@    �{�@    ��@    �$�@     8�@     �@     �@     ��@     m�@     ط@     m�@     >�@     ��@     s�@     \�@     .�@     ҩ@     ��@     ��@     ��@     �@     \�@     ��@     |�@     T�@     t�@     ��@     �@     �@      �@     x�@     Ќ@     p�@     ؍@     �@     @{@      {@     �w@      z@     �r@     �o@     �q@     �h@     0r@     @r@     `h@     `g@     `v@     �f@      [@      d@      X@     �U@     �V@      R@     �H@      X@     �m@     �H@     �K@     �L@      A@      A@     �i@     �H@      ?@      @@      *@      1@      2@      2@      3@      8@      1@      @      $@      &@      @      @       @       @      P@      @      @      @      @      @      @      �?      �?      @       @      �?      @      �?      �?      @              @      �?               @      �?      @      �?              �?      �?              �?              �?              �?              �?       @      �?               @      �?      �?      @      �?      @      �?      �?      �?      �?       @       @      �?      @      �?      @      @      @      @      @      @      @      @      @      @      (@      $@      @      ,@      @      $@      7@     �J@      3@      7@     �@@      4@      :@      @@      N@     �K@     �\@     �D@      H@     �K@     �S@     @R@     �N@     �R@     @T@     ``@     �Z@      \@     0s@      a@     �e@      d@     �k@     �h@     q@      w@     �@     @u@     0x@     �w@     �w@     ��@     h�@     ��@      �@     ��@     ��@      �@     ��@     `�@     ��@     �@     4�@     ܙ@     ��@     ��@     ��@     f�@     ��@     ��@     ة@     L�@      �@     ̱@     X�@     �@     (�@     Ѹ@     �@     �@     ��@    ���@    ���@     
�@     x�@    �l�@    ��@    ��@     5�@    ���@    ���@     ��@    ���@     ��@    �G�@     ��@    ���@     �@     ��@     =�@     ��@     ��@     �@     ��@     ��@     ܔ@     8�@     �v@      b@      M@      5@      @        
�
Conv2/h_conv2*�    O��?     $#A! �H|1�@)��iP��@2�        �-���q=�5�L�>;9��R�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             cA              �?              �?       @      �?               @      �?      �?      @      �?      @      �?      �?      �?      �?       @       @      �?      @      �?      @      @      @      @      @      @      @      @      @      @      (@      $@      @      ,@      @      $@      7@     �J@      3@      7@     �@@      4@      :@      @@      N@     �K@     �\@     �D@      H@     �K@     �S@     @R@     �N@     �R@     @T@     ``@     �Z@      \@     0s@      a@     �e@      d@     �k@     �h@     q@      w@     �@     @u@     0x@     �w@     �w@     ��@     h�@     ��@      �@     ��@     ��@      �@     ��@     `�@     ��@     �@     4�@     ܙ@     ��@     ��@     ��@     f�@     ��@     ��@     ة@     L�@      �@     ̱@     X�@     �@     (�@     Ѹ@     �@     �@     ��@    ���@    ���@     
�@     x�@    �l�@    ��@    ��@     5�@    ���@    ���@     ��@    ���@     ��@    �G�@     ��@    ���@     �@     ��@     =�@     ��@     ��@     �@     ��@     ��@     ܔ@     8�@     �v@      b@      M@      5@      @        

cross_entropy_scl�<�>

training_accuracy=
W?��"$S      �:
A	6$�b���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�`����L����mKe-!9��*h���������A!�I-��763,U�y�?�F��^�j��cuE�W&��LJ�!�TD!	w��p��L�O�:�b@�?d�8M'ë��6��ߌ�Y�=I��ׯ��RP�����Țl�yi�j�z������"ɄU�Ba��n�m(����P��\�DYYYYD2������$ ]����Bt������^Ύ�2�o��,�"EM  �:h�V�K�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(���/KCQ ���)\YpZ� *W����j�X-���I�8��M�AS�A��
ód۹ܻa�m��^�Á����.�Xiwn�|�.PQ�Bٰ�QSofC��ŹK��X-�pGu<��/�u�hQ}�B<V[{ �����X��Y:
�?�6L�y� ���%Y����c�,5��v0=	��#�R��ur�z�9۪Sٶ���'�����;�R��!��:��������t��ڰm    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  IDAT(�ő�JQ��b����4R(Q�K
�V>A���t�'�$h)�k�\H!V�)��E@!(�a9���l���6��ǜ�a���0���%u�k�HRȆ�d ��[��5 o��J&�^bWۿwP�̧��w^� ^i�)� ٜ`�srR3���S!���}�2�����s�
��x쿶�����mǵH�2q71/K�������Zųkm�(�D���+$Ǳ|;
�} ��u�n)XS X^��/��n8PJ��ƛ�C�}gvlܖ    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H  IDAT(�c`,�� 'N9�W��`:y�pI)4�����N��"�I=���,z(�����?I[��%Ad9�{N��8��VBq��<S[�ƿ,�IL��g~��1h1�k����ɦ�������߿����{΋�Yn� %��p�8�4��*��c͟�R&ߟ}d``���p��9�������T�;ga```(�`���*��pf#�G86�����~��d����;���d,�|�
a0a�����d������t  .bT�l�    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�`�5�R��~��d`��{P!�c�03�00\�c�E6�`�"�������p�A��N�[100�����C��o0�� ������ǯc�����������k�)�`���߿�Q$^y~������+#��Iݼz�9؍��J����p��}�	�������fF���w))N� �;�$���    IEND�B`�
%
Conv1/weights/summaries/mean_1�un�
'
 Conv1/weights/summaries/stddev_1s��=
$
Conv1/weights/summaries/max_1�J>
$
Conv1/weights/summaries/min_1��P�
�
!Conv1/weights/summaries/histogram*�	    Vʿ    �S�?      �@!  Fn�I�)�:���|@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�k�1^�sO�IcD���L���%�V6��u�w74��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���%�V6?uܬ�@8?�T���C?a�$��{E?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @      @      ,@      0@      0@      3@      ;@      0@      2@      7@      3@      1@      ,@      .@      ,@      2@      4@      0@      3@      "@      (@      "@      "@      @      @      @      @      @      @      @      @      @       @              @       @      �?      @       @       @       @      �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              @      �?      �?               @      @      �?      @      @      @              �?      @      @      @      @      @       @      @      @      (@      @      "@      "@      $@      @      $@      ,@      (@      0@      2@      0@      7@      1@      (@      (@      5@      $@      6@      ,@      .@      1@      (@        
$
Conv1/biases/summaries/mean_1��=
&
Conv1/biases/summaries/stddev_1�=C;
#
Conv1/biases/summaries/max_1`�=
#
Conv1/biases/summaries/min_1P�=
�
 Conv1/biases/summaries/histogram*q	   @ 
�?   ���?      @@!   g]	@)���!���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               <@      @        
�!
Conv1/conv1_wx_b*�!	   �Z��   ����?     $3A! ���}9�@)�K#I��@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿp��Dp�@�����W_>�;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�               @     �V@     �x@     @�@     ؎@     |�@     �@     �@     Z�@     H�@     ��@     �@     :�@     ��@     ��@     ��@     ��@     �@     t�@     g�@     ��@     ��@     7�@     j�@     W�@     Y�@     ĳ@     ~�@      �@     �@     W�@     Ю@     ��@     ��@     v�@     ��@     �@     .�@     R�@     H�@      �@     ,�@     ��@     �@     ��@     |�@     x�@     L�@     `�@     0�@     H�@     ��@     ��@     X�@     ��@     H�@     �|@     @y@     Px@     x@     0s@     �s@      q@      n@     �m@     `n@     �d@      i@     �b@     �`@     `a@     @_@     �V@     �Z@     �T@     �P@     �T@     @Q@      M@     �N@      E@      D@     �B@      @@      >@      ?@      >@      A@      4@      6@      4@      &@      7@      0@      .@      (@      0@      "@      ,@      @      @      @      @      @      @       @      @      @      �?       @       @      @       @      �?      @       @      �?      @      @      @              �?       @       @      �?              �?              �?      �?              �?      �?              �?              �?              �?      �?       @               @              �?              @               @      �?      @      @      @      @      �?              �?              @      @       @      @      @      $@      (@       @      ,@      "@      ,@      (@      3@      1@      *@      3@      1@      3@      :@      >@      A@     �@@      D@      @@      ?@     �J@      Q@      J@     @P@      O@     @S@     �T@     @S@      Z@     �Z@     �]@     �a@     @b@     �c@     �k@     �e@     `i@     �m@     p@     Pp@      s@     �u@     �v@     �x@     P}@      �@     ؁@     ��@     ��@     ��@     ��@      �@     H�@      �@     ��@     $�@     �@     ��@     ��@     "�@     ��@     Z�@     z�@     Z�@     �@     Z�@     6�@     s�@     �@     f�@    �"�@    ��@     ��@    ��@    ��"A    p��@    @g�@     ��@     ��@    ���@     ��@    ���@     ^�@     �@     h�@    ���@     w�@     �@    ���@    �G�@     [�@     A�@     ʳ@     p�@     �@     ��@     ��@     �@     �u@     �Y@      ,@      @        
�
Conv1/h_conv1*�   ����?     $3A! 5IDA)��+��@2�        �-���q=;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            ��A              �?              �?      �?       @               @              �?              @               @      �?      @      @      @      @      �?              �?              @      @       @      @      @      $@      (@       @      ,@      "@      ,@      (@      3@      1@      *@      3@      1@      3@      :@      >@      A@     �@@      D@      @@      ?@     �J@      Q@      J@     @P@      O@     @S@     �T@     @S@      Z@     �Z@     �]@     �a@     @b@     �c@     �k@     �e@     `i@     �m@     p@     Pp@      s@     �u@     �v@     �x@     P}@      �@     ؁@     ��@     ��@     ��@     ��@      �@     H�@      �@     ��@     $�@     �@     ��@     ��@     "�@     ��@     Z�@     z�@     Z�@     �@     Z�@     6�@     s�@     �@     f�@    �"�@    ��@     ��@    ��@    ��"A    p��@    @g�@     ��@     ��@    ���@     ��@    ���@     ^�@     �@     h�@    ���@     w�@     �@    ���@    �G�@     [�@     A�@     ʳ@     p�@     �@     ��@     ��@     �@     �u@     �Y@      ,@      @        
%
Conv2/weights/summaries/mean_1����
'
 Conv2/weights/summaries/stddev_1!��=
$
Conv2/weights/summaries/max_1�R>
$
Conv2/weights/summaries/min_1��T�
�
!Conv2/weights/summaries/histogram*�	   ���ʿ   ��C�?      �@! UE�J�)��b
E�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �I��P=��pz�w�7��})�l a��ߊ4F��jqs&\�ѾK+�E��Ͼ�XQ�þ��~��¾��~]�[�>��>M|K�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �H@     ��@     8�@     8�@     ȏ@     ��@     �@     ��@     `�@     p�@     ��@     |�@     �@     Ў@     x�@      �@     h�@     Ј@     ؅@     ��@     ��@     ��@     p~@     0|@     x@     �x@     �w@     �t@     �r@     �p@     @m@     �m@     �i@     �e@      b@     �a@     �\@     @`@     �\@      \@     �W@     �Y@     �T@     �Q@     �O@     �M@      E@     �I@      B@      B@     �E@     �A@      ?@      9@      6@      3@      5@      4@      5@      ,@      (@       @      $@      .@      ,@      &@      0@      @      @      @      @      "@      @      �?       @      @      �?      @      @       @      �?      @       @      @       @              �?      �?              �?              �?              @              �?              �?              �?              �?              �?              �?      �?       @               @       @      @      @      @       @      �?      �?      �?      �?      @      @       @      @      �?      @      @      @      @      @      @      "@      ,@      &@       @       @      (@      ,@      8@      6@      4@      (@      ?@      5@      B@     �D@      <@     �I@     �H@     �C@      E@     �D@      N@     @S@     @S@     �S@      V@      W@      ^@     �[@     �`@     @a@     �`@      d@     �h@     @j@     �l@      m@     �p@     �p@     �t@     �t@     @x@     @y@     ~@     �@     h�@     Ё@     �@     І@     �@     p�@     ��@     (�@     �@     ��@     ��@     P�@     ܒ@     �@     |�@     x�@     `�@     ��@     h�@     �@     H�@      8@        
$
Conv2/biases/summaries/mean_1��=
&
Conv2/biases/summaries/stddev_1a�:;
#
Conv2/biases/summaries/max_1]�=
#
Conv2/biases/summaries/min_1��=
�
 Conv2/biases/summaries/histogram*q	    ���?   ��?      P@!  �� 5@)�w�P���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:              �M@      @        
�#
Conv2/conv2_wx_b*�#	   `�t��   ��=�?     $#A! �h����@)O'����@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ�XQ�þ��~��¾�*��ڽ�G&�$��R%������39W$:�����z!�?��T�L<��BvŐ�r>�H5�8�t>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              @      ;@     @\@     �i@     �z@     ��@     D�@     ��@     ��@     �@     ��@     ��@     ��@     2�@     4�@    ���@     �@    ���@     #�@    ���@    ���@    ���@    �c�@     m�@    ���@     ��@    �X�@     ��@     ��@    ��@     �@     ��@     ��@     ��@     z�@     ε@     ��@     �@     �@     R�@     ��@     t�@     :�@     "�@     ��@     ܡ@     ޠ@     �@     ��@     �@     T�@     ��@     x�@     ��@     �@     ؏@     ��@     ��@     Ȃ@     �@     P�@     `@     �{@     @y@     0{@     �u@     �v@     �p@     �o@     �p@     �l@     �v@      b@     @`@     @]@      c@     @]@     �W@     �X@     �X@      P@     �M@     @Q@      F@     �I@     �F@     �L@      B@      D@     �Y@      >@      8@      0@      4@     �G@      6@      (@      $@      $@      (@      *@      &@      @      &@      @      @      @      @      "@      @      @      �?      @      @       @              @      @      @              �?              �?       @              �?              �?              �?      �?               @               @              �?              �?              �?              �?              �?              �?               @              �?              �?              �?       @              �?      @              @      �?      @      @      @      @      @      @      @      @      "@      @      @      @      $@      @      F@       @      &@      (@      *@      6@      4@      6@      7@      B@      ,@      @@     �B@      k@      :@     �T@     �H@      M@      I@      N@      [@      ]@     �R@     @q@     �Y@     �X@     �j@     �]@     �d@     �h@      g@     @f@     �f@     �o@     �q@     �w@     �u@     �x@     �z@     @�@      {@      �@     ��@     ��@     ��@     �@     ��@     `�@      �@     �@     p�@     ��@     ��@     ܙ@     @�@     ��@     ��@     �@     &�@     R�@     p�@     H�@     ��@     ֱ@     \�@     ��@     ��@     E�@     �@     ��@     ��@     ��@     c�@     w�@    ���@    ��@    ���@     e�@    ���@    ���@     M�@    ���@    �F�@     �@    ���@     ��@    ���@     ��@     �@     ��@     ��@     ��@     ��@      �@      �@     ��@     y@     `g@      L@      @@      @      �?        
�
Conv2/h_conv2*�   ��=�?     $#A! �;Q���@)V��c��@2�        �-���q=BvŐ�r>�H5�8�t>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�            ��A              �?              �?               @              �?              �?              �?       @              �?      @              @      �?      @      @      @      @      @      @      @      @      "@      @      @      @      $@      @      F@       @      &@      (@      *@      6@      4@      6@      7@      B@      ,@      @@     �B@      k@      :@     �T@     �H@      M@      I@      N@      [@      ]@     �R@     @q@     �Y@     �X@     �j@     �]@     �d@     �h@      g@     @f@     �f@     �o@     �q@     �w@     �u@     �x@     �z@     @�@      {@      �@     ��@     ��@     ��@     �@     ��@     `�@      �@     �@     p�@     ��@     ��@     ܙ@     @�@     ��@     ��@     �@     &�@     R�@     p�@     H�@     ��@     ֱ@     \�@     ��@     ��@     E�@     �@     ��@     ��@     ��@     c�@     w�@    ���@    ��@    ���@     e�@    ���@    ���@     M�@    ���@    �F�@     �@    ���@     ��@    ���@     ��@     �@     ��@     ��@     ��@     ��@      �@      �@     ��@     y@     `g@      L@      @@      @      �?        

cross_entropy_scl�%�>

training_accuracyfff?D[�R      ��b�	<he���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`��( �ħ����300� K�&�E�030����F$9���3��=��@�3�~��-�ߗTf8�p-Ô+Qv/����'q�ߏ�����w_S2�߿NG��w�C����2G����1$��iB��=��
����@�L�~O������%C'?3�F�����C�wڟ�ٰJ1�����&�G��8X�l'����6yt-�����߿�;ܘ��Ro��wyY�NwR  �%]�"�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`������:��Ȱ<���￿g`������D=���o����>�=���4��߿l�Y��108=���^��]�
����������>����3Vo������)l�```P���_� v9���#0!��,>84f�����6�r6������O��FW�1�ǪQ��ww�����=�]�XFF��Ǐ�J�W��.���{lvz|����߳�ؽrg�
W�RT t�`0&?    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�?`F�/p	�b�;�YqJ��׋SN��?S��6���#xL����>�d�g8��T������:E��)�Ĉ�Es���8%9U������Xܒn�~�4`8}�$�,+���K��xt���
� ��pY f    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�؅9�>�J&l�����$�ag>���XX������X��%t�������uh"����%�bO~#I��Ym._�ۇ�L��+��'c���8ˑ0l�P~F	���ɋLXe$tKw��eAS�7�8���k�  ��)'��S    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H  IDAT(�c`J��'���1e��������%�T������c��t� C�������_����p����]����S��������}y�#�"����_9��t�ԋOpxkѿ�\p��/�����_?�i���������j�����'��~�a3�͖w�`"��r0���Ϡ���'TY����h����d����z����Ҟ�۴���.as�����[R�	`D��Tex��'� ?�a���:    IEND�B`�
%
Conv1/weights/summaries/mean_1�$k�
'
 Conv1/weights/summaries/stddev_1o��=
$
Conv1/weights/summaries/max_1�	K>
$
Conv1/weights/summaries/min_1�P�
�
!Conv1/weights/summaries/histogram*�	   `Cʿ    1a�?      �@! 0�Қ��)hr�۠�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed���bB�SY�ܗ�SsW�8K�ߝ�>�h���`�>��VlQ.?��bȬ�0?��82?�u�w74?�T���C?a�$��{E?
����G?�qU���I?ܗ�SsW?��bB�SY?�l�P�`?���%��b?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      *@      0@      0@      4@      :@      2@      1@      7@      4@      .@      *@      1@      *@      1@      2@      0@      5@      &@      &@      &@      @       @      @      @       @      @      @      @      @      @      @       @              @      �?       @       @       @       @       @              �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?               @              �?              @      �?      @       @      @              �?      @       @      �?      �?      @      @      @      @      @       @      @       @      "@       @       @      &@      "@      @      &@      ,@      *@      0@      1@      0@      6@      2@      (@      *@      4@      $@      6@      ,@      0@      0@      (@        
$
Conv1/biases/summaries/mean_1�^�=
&
Conv1/biases/summaries/stddev_1s;G;
#
Conv1/biases/summaries/max_1� �=
#
Conv1/biases/summaries/min_1}(�=
�
 Conv1/biases/summaries/histogram*q	   ��?    ��?      @@!   ��	@)GAY��?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               <@      @        
�!
Conv1/conv1_wx_b*�!	   �w��   @�?�?     $3A! M3��e�@)�$?%lj�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;0�6�/n���u`P+d��;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              �?      `@     �~@     ��@     4�@     @�@     L�@     ��@     ��@      �@     ک@     <�@     N�@     ��@     <�@     ��@     M�@     ��@     ^�@     µ@     $�@     }�@     ��@     ϵ@     x�@     ��@     ˲@     ?�@     N�@     �@     h�@     ��@     ��@     ��@     D�@     x�@     J�@     ��@     d�@     ڠ@     ��@     �@     @�@     ؗ@     �@     L�@     ��@     �@     ��@     ��@     ��@     І@      �@      �@     ��@     �~@     @z@     �x@     �x@     �v@      s@      q@     �n@     �o@     �j@     �h@     �e@     `d@      a@     `b@     �[@     �[@     @[@     �U@     �W@     �S@      W@      O@     �H@      Q@      B@     �B@      B@      ?@     �@@      6@      :@      :@      <@      7@      3@      "@      &@      (@      1@      0@       @      &@      @      @       @      @      @       @      @      �?      @      @      @      @       @      @       @              �?      �?      @      �?               @      @              �?              �?              �?              �?              �?              �?               @              �?              �?              �?       @               @              �?       @       @              @      �?      @       @      @       @      @      @      @       @       @      @       @       @       @      &@      (@      @      (@      .@      0@      &@      7@      ,@      5@      =@      6@      8@      >@      @@      G@      F@      G@      E@     �K@      J@      R@     �S@      T@     @W@     @Y@     @\@      `@     �a@     @b@      b@     �g@      k@     �k@     �l@     r@     `q@     �v@     @v@     �w@     0}@      ~@      �@     P�@     ��@     p�@     �@     ��@     d�@     �@     X�@     D�@     �@     �@     �@     2�@     V�@     ��@     ��@     `�@     ��@     ��@     F�@     C�@     �@     ��@     <�@     �@    ���@    @l�@    |�#A     ~�@    @�@    ���@     ��@    ���@    ���@     v�@    ���@    ��@     ��@    �>�@     ��@    ���@     ^�@     q�@     j�@     
�@     γ@     Ԯ@     �@     Ğ@     P�@     P�@     �x@      ]@      :@      @        
�
Conv1/h_conv1*�   @�?�?     $3A!���:�-A)'��:+Q�@2�        �-���q=;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            ��A              �?              �?               @              �?              �?              �?       @               @              �?       @       @              @      �?      @       @      @       @      @      @      @       @       @      @       @       @       @      &@      (@      @      (@      .@      0@      &@      7@      ,@      5@      =@      6@      8@      >@      @@      G@      F@      G@      E@     �K@      J@      R@     �S@      T@     @W@     @Y@     @\@      `@     �a@     @b@      b@     �g@      k@     �k@     �l@     r@     `q@     �v@     @v@     �w@     0}@      ~@      �@     P�@     ��@     p�@     �@     ��@     d�@     �@     X�@     D�@     �@     �@     �@     2�@     V�@     ��@     ��@     `�@     ��@     ��@     F�@     C�@     �@     ��@     <�@     �@    ���@    @l�@    |�#A     ~�@    @�@    ���@     ��@    ���@    ���@     v�@    ���@    ��@     ��@    �>�@     ��@    ���@     ^�@     q�@     j�@     
�@     γ@     Ԯ@     �@     Ğ@     P�@     P�@     �x@      ]@      :@      @        
%
Conv2/weights/summaries/mean_1����
'
 Conv2/weights/summaries/stddev_1���=
$
Conv2/weights/summaries/max_1�R>
$
Conv2/weights/summaries/min_1CRU�
�
!Conv2/weights/summaries/histogram*�	   `H�ʿ    W�?      �@! p�J�)�7��	�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �I��P=��pz�w�7���ߊ4F��h���`���(��澢f����
�/eq
Ⱦ����žK+�E���>jqs&\��>���%�>�uE����>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �J@     H�@     `�@     �@     �@     ��@     �@     ��@     <�@     |�@     ��@     ��@     ̐@     ؎@     0�@     @�@     `�@     0�@     ��@     ؃@     ��@     ؀@     0~@      |@     `x@     �y@     �v@     Pt@     pr@     �o@     pp@     �k@     �j@      f@     �a@     @a@      `@     �]@     @]@     �X@     @Z@     @Z@     @U@     �R@      P@      H@      K@      F@      B@      D@      C@      9@     �@@      8@      ;@      6@      5@      5@      5@      .@      3@      (@      ,@      "@      @      &@       @      @       @      $@      @      @      �?      "@      @      @      @       @      �?      �?      �?      �?      �?               @      �?       @       @       @      �?              �?       @              �?              �?              �?              �?              �?              �?              �?               @               @              �?      @      �?      �?       @      �?      @              �?      @      �?       @               @      @      @      @      @      @      @       @      $@      @       @      @      @      $@      ,@      $@      (@      7@      5@      <@      ;@      8@      ?@      >@      =@     �A@     �A@      >@      K@      F@     �F@     �O@      S@     @S@     �S@     �U@     �Y@     @Z@      [@     �`@     �`@     `a@      e@     �h@      i@      n@     �m@     `o@     �r@     Ps@     0u@     �w@     �y@     @     (�@     P@     ��@     `�@     ��@     ��@     ��@     0�@     ��@     ؎@     ��@     �@     8�@     ��@     4�@     `�@     ��@     d�@     P�@     ��@     ��@     H�@      <@        
$
Conv2/biases/summaries/mean_1B��=
&
Conv2/biases/summaries/stddev_16	@;
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1���=
�
 Conv2/biases/summaries/histogram*q	   ����?   @�X�?      P@!  �34@)LK��!��?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               M@      @        
�#
Conv2/conv2_wx_b*�#	   ��I��   �G�?     $#A! �����@)z�sz���@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f����E��a�Wܾ�iD*L�پ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�XQ�þ��~��¾G&�$��5�"�g����u��gr�>�MZ��K�>����>豪}0ڰ>G&�$�>�*��ڽ>�[�=�k�>��~���>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              @      <@      T@     �f@     @y@     ��@     ��@     ��@     |�@     B�@     *�@     w�@     ��@     ��@     �@     ��@    ���@     ~�@     `�@     ��@    ���@     ��@     ��@    ���@     ��@    ���@    ���@    ���@     Q�@     ��@     �@     ��@     �@     �@     ��@     w�@     ,�@     ��@     (�@     ��@     �@     X�@     ��@     @�@     v�@     ��@     ��@     ؞@     �@     �@     ��@     @�@     ��@     ̔@     x�@     Ќ@      �@     �@     P�@     ��@     ��@     p}@     (�@      z@     ��@     �}@     �v@     ؄@     p|@     �h@      m@     �c@     �b@     �`@     �a@     @\@     `e@     `c@      V@      W@      L@     �M@     �v@      K@     �P@      C@      O@      9@      D@      <@      :@      1@     �M@      5@     �Q@      ,@      4@      3@      H@      $@      ,@      $@      "@      @      $@      @      @      "@      @      @      @       @      @      �?      @      �?      @      �?              �?      �?       @      @              @      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?      �?      �?       @      @      @      �?              �?      �?       @     �O@      @       @       @      I@      @      "@      @      @      @      @      @      "@      "@      ,@      ,@      1@      .@      (@      ?@      8@      2@      8@      5@      :@     �H@      :@      <@      N@      M@     �R@     �Q@     �N@     �Z@     �o@      U@     �S@      ]@     @[@     �d@     �e@     �g@     �t@      p@     �l@     �n@     �q@     �q@      |@     h�@      ~@     X�@     �@      �@     �@     @�@      �@     �@     x�@     �@     (�@     ̕@     ��@     0�@     �@     ��@     ��@      �@     J�@     j�@     4�@     ڬ@     �@     y�@     |�@     R�@     �@     0�@     �@     -�@     W�@     ��@     I�@    ���@     ��@     ��@    �_�@    ���@    ���@     ��@     ��@     ��@    �X�@     L�@    �9�@     �@     ��@     ��@    ���@    ���@     =�@     M�@     ��@     �@     (�@     ��@     Ĕ@      �@     @w@      i@      M@      7@      @        
�
Conv2/h_conv2*�   �G�?     $#A! j\~�&�@)��>}��@2�        �-���q=�u��gr�>�MZ��K�>����>豪}0ڰ>G&�$�>�*��ڽ>�[�=�k�>��~���>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�            �sA              �?              �?              �?              �?              �?              �?       @              �?      �?      �?      �?       @      @      @      �?              �?      �?       @     �O@      @       @       @      I@      @      "@      @      @      @      @      @      "@      "@      ,@      ,@      1@      .@      (@      ?@      8@      2@      8@      5@      :@     �H@      :@      <@      N@      M@     �R@     �Q@     �N@     �Z@     �o@      U@     �S@      ]@     @[@     �d@     �e@     �g@     �t@      p@     �l@     �n@     �q@     �q@      |@     h�@      ~@     X�@     �@      �@     �@     @�@      �@     �@     x�@     �@     (�@     ̕@     ��@     0�@     �@     ��@     ��@      �@     J�@     j�@     4�@     ڬ@     �@     y�@     |�@     R�@     �@     0�@     �@     -�@     W�@     ��@     I�@    ���@     ��@     ��@    �_�@    ���@    ���@     ��@     ��@     ��@    �X�@     L�@    �9�@     �@     ��@     ��@    ���@    ���@     =�@     M�@     ��@     �@     (�@     ��@     Ĕ@      �@     @w@      i@      M@      7@      @        

cross_entropy_scl�j�>

training_accuracy�(\?�㮰T      ~��	�{Ig���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   IDAT(��Ϳ/q�M,nA#�8'1I$1���F���6�Zl��b�`��z)���!a2��'�ӓ��5h��Sk��y���?-AL�3i������$Y�c���*?/����+3�V�{��o�0^�i ��k_�%|�+q�'�5�L��<������  C,�������2�/"Iw(h+J�
������OwUd��w
5���EҎ
��9(]Gd.*�bu�D&�/�m>�N �ɼՁv���7=+�H'{Lһ�0�m��^���V>4I5&l��4v�O�v
�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H  IDAT(�c`d��Pc`�������������)���=6��"���x�:e�������0��w/\��Ó��&�?�����p�a,õ�k7P����wݵLp[ N/������{�/����)�	���ӎ;�~Ȓ��;��߿��U�x���q~\�����	\�������a���I���������.>�������*������5��9	+30�t{���꯿Pp�Z.
����+v=��T�  Y�W����    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c`܀�����Dl2,*�'�����6Iÿ�DS���߿��ݾ-SN��߿U�3bJ�My����������߿��P����a````X������s����0�(3��%w���Ǐy9���o=�f��u��?��auC굿+qJ2h��)a���j�
e}eP�B�Ԏt��1B�ȿ�u�
���G�)����yNi�����#���B�0�#OA��c����o��p��"�  �s��.�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�5���/�\ϗ������$�gqs�o��AiO6���>aН��~���߿�����Ĕe�򷈙��y埿v�Z~���������~,�:m���.CVI.�%��;c���; &�j�J�0��)� ����t��e��ZuZ���/��w<��L2}�|�Q@�����'�=W��	�PI9E&�����ol���j�'d��{������ϛ�H��p���öW�0�6�  �m��~K    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`6����h�� Y��Ć��Q����_(x�
d�JV�1
s3000\g``��E�๿�?nk+���kTS}��}����� ���*�tk�߿�0Q&pz�ł&#����M��y"7��߿���A �^<���.B؄���H"�%1����/�l��ph�����_3�2��=)�C����j�� &�J�q���    IEND�B`�
%
Conv1/weights/summaries/mean_1xab�
'
 Conv1/weights/summaries/stddev_1�Ҷ=
$
Conv1/weights/summaries/max_1*�K>
$
Conv1/weights/summaries/min_1�FQ�
�
!Conv1/weights/summaries/histogram*�	   ��(ʿ   @�u�?      �@!  M��)�ɍ���@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�<DKc��T��lDZrS�uܬ�@8���%�V6��u�w74?��%�V6?�qU���I?IcD���L?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      *@      .@      0@      5@      :@      2@      1@      7@      4@      0@      &@      2@      (@      4@      .@      .@      6@      (@      &@      $@      $@      @      @      @      @      "@       @      @      @       @      @              �?              @      �?      @       @              @      �?       @      �?              �?              �?              �?              �?              �?              �?               @      �?              �?              �?       @      @      �?      �?      �?       @      @              �?      @              @      @      @      @      @      @      �?      @      @      "@       @       @      (@      $@      @      &@      .@      &@      0@      3@      ,@      7@      2@      &@      0@      2@      $@      6@      .@      ,@      2@      &@        
$
Conv1/biases/summaries/mean_1e=�=
&
Conv1/biases/summaries/stddev_1N�N;
#
Conv1/biases/summaries/max_1�%�=
#
Conv1/biases/summaries/min_1%�=
�
 Conv1/biases/summaries/histogram*q	    ��?    ���?      @@!   ��	@)z��5��?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               <@      @        
�!
Conv1/conv1_wx_b*�!	   ���    ���?     $3A! ���&�@)�n@�L�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(�������ž�XQ�þ5�"�g���0�6�/n���MZ��K���u��gr���
�%W�>���m!#�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @     �c@     �}@     ��@     ��@     �@     �@     ܣ@     b�@     T�@     ��@     B�@     o�@     �@     ��@     ��@     µ@     �@     .�@     f�@     �@     ��@     S�@     ��@     ޵@     5�@     @�@     ��@     ��@     1�@     ?�@     �@     ��@     \�@     ��@     f�@     P�@     X�@     8�@      �@     d�@     x�@     <�@      �@     L�@     ē@     `�@     �@     ��@     ��@     ��@     ȇ@      �@     �@     Ё@     �@     8�@     p|@     �w@     �u@     �v@     �r@      o@     �m@     �n@     �j@      f@     �f@     �c@      b@     @`@     �Y@      ]@     �V@      W@      Q@     @R@      S@     �R@     �K@     �I@     �G@      E@      @@     �E@     �I@      2@      ?@      6@      9@      6@      (@      1@      3@      1@      1@      (@       @      (@      "@      @      @      $@      $@      @       @       @      "@       @      @      �?      @      @       @      @       @      �?       @      �?              �?      �?              �?              �?              �?              �?               @              �?              �?              �?      �?              �?      �?              �?              @               @      @       @      @      @      �?       @      @       @      @      @      �?      @       @      "@      @      @      "@      "@      &@      @      &@      ,@      1@      2@      3@      1@      ,@      3@      >@      6@      8@      >@     �@@      H@     �A@      E@      J@      I@     �K@      T@      T@     @Q@     �T@      X@     @\@     ``@     �^@      d@     `e@     @f@     �f@     �j@     �m@     `q@     �p@     `s@     @u@     y@     �{@     �}@     ��@     (�@     ��@     h�@     P�@      �@     ��@     �@     ��@     ��@     ܖ@     ��@     �@     @�@     h�@     ��@     D�@     v�@     P�@     x�@     }�@     ��@     �@     ��@     7�@    ��@    ���@    ���@     K�@    ��"A    p��@    ���@     \�@    �P�@     '�@    �q�@    ���@    ���@    ��@     ��@     �@     j�@     l�@     .�@    ��@     ��@     v�@     ŵ@     α@     �@     ~�@     0�@     @�@     `~@     @c@     �E@      "@        
�
Conv1/h_conv1*�    ���?     $3A!���i{A)���}�@2�        �-���q=�
�%W�>���m!#�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            �XA              �?               @              �?              �?              �?      �?              �?      �?              �?              @               @      @       @      @      @      �?       @      @       @      @      @      �?      @       @      "@      @      @      "@      "@      &@      @      &@      ,@      1@      2@      3@      1@      ,@      3@      >@      6@      8@      >@     �@@      H@     �A@      E@      J@      I@     �K@      T@      T@     @Q@     �T@      X@     @\@     ``@     �^@      d@     `e@     @f@     �f@     �j@     �m@     `q@     �p@     `s@     @u@     y@     �{@     �}@     ��@     (�@     ��@     h�@     P�@      �@     ��@     �@     ��@     ��@     ܖ@     ��@     �@     @�@     h�@     ��@     D�@     v�@     P�@     x�@     }�@     ��@     �@     ��@     7�@    ��@    ���@    ���@     K�@    ��"A    p��@    ���@     \�@    �P�@     '�@    �q�@    ���@    ���@    ��@     ��@     �@     j�@     l�@     .�@    ��@     ��@     v�@     ŵ@     α@     �@     ~�@     0�@     @�@     `~@     @c@     �E@      "@        
%
Conv2/weights/summaries/mean_1�?��
'
 Conv2/weights/summaries/stddev_1���=
$
Conv2/weights/summaries/max_1TS>
$
Conv2/weights/summaries/min_1��U�
�
!Conv2/weights/summaries/histogram*�	    3�ʿ   ��j�?      �@! ���i2K�)���0�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���pz�w�7��})�l a��ߊ4F����(��澢f����0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>['�?��>K+�E���>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �I@     ��@     0�@     (�@     �@     ��@      �@     t�@     $�@     ȓ@     X�@     В@     ��@     �@     h�@     ȍ@     ��@     h�@     ��@     0�@     P�@     ��@      ~@     �{@     Px@     0y@     pv@     0t@     �r@     Pp@     �p@     �k@     @j@      f@     �a@     �a@     �a@     �]@     @Z@     @Y@      Y@     �Y@      W@      R@     �L@     �G@     �H@      G@      F@      A@     �B@     �A@      8@      8@      =@      <@      4@      1@      8@      0@      0@      (@      @      "@      ,@      $@      @      @      @      $@      @               @      @      @      @      @      @      �?      @      @      @      @              �?       @      �?               @               @              �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?              �?      �?              @      �?              @              @      @      @      �?      �?       @      �?      @      (@      @      @      @       @      $@      @      $@      "@      "@      &@       @      .@      0@      6@      :@      .@      8@      3@      <@      <@     �D@      ;@      F@     �F@      F@     �J@      G@     �O@      Q@     �S@     �R@     �R@      T@     @]@     �^@      `@     @a@     ``@     �e@      i@      i@     �o@     �k@     �o@     �r@     �s@     0u@     0x@     �z@     �|@     p�@     �@     x�@     @�@     ��@     H�@     p�@     �@     @�@     ��@     ��@     �@     �@     ��@     h�@     0�@     ��@     ��@     H�@     ��@     Љ@     8�@      <@        
$
Conv2/biases/summaries/mean_1��=
&
Conv2/biases/summaries/stddev_1r�D;
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1r��=
�
 Conv2/biases/summaries/histogram*q	   @.�?   `<Z�?      P@!  ���/@)���'���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:              �L@      @        
�$
Conv2/conv2_wx_b*�$	   �b� �    ,g�?     $#A! ��uwR�@)��q 8i�@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ
�/eq
Ⱦ����ž�u`P+d����n�����X$�z��
�}����f^��`{>�����~>�MZ��K�>��|�~�>;9��R�>���?�ګ>�*��ڽ>�[�=�k�>�XQ��>�����>
�/eq
�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              �?      (@     �D@     @[@     �l@      }@     P�@     �@     ��@     ��@     .�@     ��@     ʶ@     ¹@     �@    ��@    ���@    ��@    ���@    ���@    ���@    �9�@    ���@     O�@    �c�@     ��@    ��@    �2�@    ���@    ���@     2�@     v�@     S�@     ��@     Z�@     ]�@     W�@     ��@     �@     ��@     �@     ��@     �@     2�@     �@     R�@     f�@     :�@     t�@     ȝ@     d�@     ��@     t�@     ��@     P�@     ��@     ��@     ��@     (�@     �@     ��@      �@     P|@     �z@     �w@     u@      w@     �q@      l@      n@     @e@     �h@     �e@     �q@      e@     �c@      `@     ``@     �^@     @X@     �X@     �R@     �K@     �L@     �P@      E@     �C@      D@      A@      ]@      :@      3@      8@      8@      9@      ,@      1@      0@      $@      ,@      1@      1@      "@      @      "@      "@      @      @      @      @      @     `a@       @      @      @      �?      @       @      �?       @      @      �?      �?              �?      �?      �?               @               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @              �?               @      �?               @       @      @       @              @      @      @      @      @      @       @      @      @      @      �?       @       @      H@      &@      "@      (@      0@      1@      6@      ,@      6@      8@      <@      =@      7@      A@      =@     �F@     �D@      H@     �L@     �R@     �R@     �V@      Z@     �U@     �a@     @]@     p@     �a@     `f@     �d@     �d@     `j@     `k@     �m@     �w@      t@     �v@      y@     �z@     �x@      @     ��@     X�@     ��@     ؇@     Б@     �@     4�@     ��@     4�@     p�@     8�@     М@     p�@     ��@     H�@     J�@     ��@     ��@     v�@     	�@     �@     ߱@     u�@     ��@     ��@     ڼ@     ��@     ��@     ܿ@    �H�@     ��@    �C�@    ���@    ��@     ��@     \�@    ��@     ��@     �@     ��@    ���@    �K�@    ��@     ��@    �D�@    ���@    �^�@     R�@     z�@     ��@     Ы@     ֢@     �@     x�@     |@      n@     �Z@      8@      "@      �?        
�
Conv2/h_conv2*�    ,g�?     $#A! ��6��@)d�?;�q�@2�	        �-���q=f^��`{>�����~>�MZ��K�>��|�~�>;9��R�>���?�ګ>�*��ڽ>�[�=�k�>�XQ��>�����>
�/eq
�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�	            ��A              �?              �?              �?              �?              �?      �?              �?              �?      �?               @              �?               @      �?               @       @      @       @              @      @      @      @      @      @       @      @      @      @      �?       @       @      H@      &@      "@      (@      0@      1@      6@      ,@      6@      8@      <@      =@      7@      A@      =@     �F@     �D@      H@     �L@     �R@     �R@     �V@      Z@     �U@     �a@     @]@     p@     �a@     `f@     �d@     �d@     `j@     `k@     �m@     �w@      t@     �v@      y@     �z@     �x@      @     ��@     X�@     ��@     ؇@     Б@     �@     4�@     ��@     4�@     p�@     8�@     М@     p�@     ��@     H�@     J�@     ��@     ��@     v�@     	�@     �@     ߱@     u�@     ��@     ��@     ڼ@     ��@     ��@     ܿ@    �H�@     ��@    �C�@    ���@    ��@     ��@     \�@    ��@     ��@     �@     ��@    ���@    �K�@    ��@     ��@    �D�@    ���@    �^�@     R�@     z�@     ��@     Ы@     ֢@     �@     x�@     |@      n@     �Z@      8@      "@      �?        

cross_entropy_scl���>

training_accuracy�Ga?4RMa�R      ȫ?	��i���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(��н
�a����n>&)���D�d1���2:����&�M
��#%��W�g�����u���"��w8j� e�yƴ���a�\@�;M1�K7�US���Lt??lEMYk�.1�
��Er�֨�Fw���ɷ�nl��� �{�i5v|�x7?n+�ĩ�������:��m_�(L�+ 5�
%	�C��� O�d]��υ    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H  IDAT(�Ő�+DQ�_��\���V�f5E��Y`#Vn�D�����V�J���X�0Q'��;)�����K����{������U�C���_�������Ɏ\��2Mq62���6(]�A��u%��7�7(/�+=��æ�R.PY�F߃�� �ԗK�`#�`ґ]3�{lxfG؁�(��hí�*��/���R��"�Roh���(�B��)Py�w:����cC�^����JA��^�� ��I;]�ȱ�*+���<���.^�|L�T���I\����	��R୸    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�Ր�JA�ǵ�"ŕ!n���*I,R�@��DS'O��tI'Z$ ؚ.��@���~bq�cW���iv������it���T��Ew��f��1�M!� U ?jՎ� ������(Y  ��V��V�g a�ԃ�]�Iii��wE��`ᐤ����H�;,!I�gۀ�ϭ?����u'r�����:r�e���j��栌�>S���aސk0�$'�cc�J�}�d�7�ppz���`    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�͏��@��&�Z1:T龃ffl>�I���H��\>:�c�=0�w$��i��o� ��0�"0��$&N�,c�:-��11]��V�(�c��ز �Xۘ�(���ujc1��,��Zx���q{�?[ �ڹ�=��Ƀ��mM��W�kL�H���BD ε�G �b;\�r�U�oX9��60�'��R��%�!    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@s΍�����^.�c�Of����5�ޗ��o�.�g�ah��
C��/\�	JK2܀0"U�0$�30000��!�c0d```(��b�������js���[�)[���߿����A�����\:�ϵ[.����_ʸ%�]�t-؉]�-��x�$
@�t`x�[R���&���X.6d���s_�  H�9��    IEND�B`�
%
Conv1/weights/summaries/mean_1Q�\�
'
 Conv1/weights/summaries/stddev_16�=
$
Conv1/weights/summaries/max_1��K>
$
Conv1/weights/summaries/min_1~Q�
�
!Conv1/weights/summaries/histogram*�	   `�/ʿ   ��}�?      �@!  N�K��)���)�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��m9�H�[���bB�SY�U�4@@�$��[^:��"��T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?<DKc��T?ܗ�SsW?��bB�SY?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      *@      .@      0@      4@      ;@      2@      2@      5@      4@      2@      $@      3@      &@      2@      1@      0@      6@      "@      ,@       @      &@      @      @       @      @      @      @      @      @      @      @      @              �?              @       @      �?       @      �?       @       @              �?      �?       @              �?              �?              �?              �?      �?              �?      �?               @      �?      �?       @       @      �?       @      �?      @      �?              @       @       @      @      @      @      @      @       @      @       @      @      @       @      *@       @      @      &@      *@      *@      1@      2@      ,@      6@      4@      "@      0@      3@      $@      5@      0@      *@      3@      &@        
$
Conv1/biases/summaries/mean_1$�=
&
Conv1/biases/summaries/stddev_1�jY;
#
Conv1/biases/summaries/max_1���=
#
Conv1/biases/summaries/min_1���=
�
 Conv1/biases/summaries/histogram*�	   @�з?   �ޱ�?      @@!   ā	@)�ȩ�Д�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @      :@      @        
�!
Conv1/conv1_wx_b*�!	   യ�   ��8�?     $3A! Lt �o�@)�
�&�/�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�K+�E��Ͼ['�?�;��n�����豪}0ڰ����?�ګ�;9��R���i����v>E'�/��x>豪}0ڰ>��n����>�[�=�k�>��~���>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              �?     �[@     �u@     @�@     ԑ@     @�@     ��@     l�@     ��@     ��@     ��@     ��@     �@     ��@     C�@     ۳@     ϵ@     �@     ��@     ��@     �@     "�@     9�@     �@     &�@     ڳ@     a�@     ֲ@     K�@     ��@     >�@     ��@     J�@     ^�@     l�@     ,�@     ��@     ��@     ��@     ̠@     ��@     0�@     �@     T�@     p�@     H�@     ��@     ,�@      �@     �@     ��@     0�@     h�@     �@     h�@     x�@     @{@     Pz@     �y@     `v@     0t@     �q@      p@     �p@     �k@     �j@     �f@      d@     �e@     �c@     `b@     �[@     �V@     @V@     �V@     �T@     �T@     �P@     �J@      L@     �M@      J@     �K@      <@     �A@      B@      ;@      8@      8@      0@      4@      <@      *@      *@       @       @      .@      "@       @       @      @      @      @      @      "@      @       @      @      �?      @       @      @      @      �?       @              �?      �?              �?              �?      @               @      �?              �?              �?              �?              �?              �?               @              �?              �?      �?              �?      �?       @      �?              @       @       @       @      �?      @      �?      @      @      @      @      @       @      (@      @      @      @      @      .@      "@      *@      (@      @      1@      7@      1@      <@      6@      9@      ;@      ;@     �C@     �A@     �E@     �E@     �E@     �E@      K@      M@     @Q@      S@     @W@     �T@      X@     `a@      a@     �a@      f@     `f@     �g@     �f@     �k@     pq@     �n@     ps@     pv@     �x@     �x@     �~@     ��@     �@     ��@     P�@     ��@     �@     0�@     L�@     ��@     ��@     ��@     4�@     �@     h�@     X�@     V�@     ��@     �@     B�@     .�@     O�@     ��@     v�@     v�@     ��@    ���@     ��@     <�@    `��@    X�!A     ��@     m�@    �K�@     ��@     ��@     2�@     ��@    ���@    �"�@     �@     %�@    ���@    �C�@     ��@    ���@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     �@     `{@     @\@      4@      @        
�
Conv1/h_conv1*�   ��8�?     $3A! ����]A)�ܮbs�@2�        �-���q=�i����v>E'�/��x>豪}0ڰ>��n����>�[�=�k�>��~���>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            H�A              �?              �?               @              �?              �?      �?              �?      �?       @      �?              @       @       @       @      �?      @      �?      @      @      @      @      @       @      (@      @      @      @      @      .@      "@      *@      (@      @      1@      7@      1@      <@      6@      9@      ;@      ;@     �C@     �A@     �E@     �E@     �E@     �E@      K@      M@     @Q@      S@     @W@     �T@      X@     `a@      a@     �a@      f@     `f@     �g@     �f@     �k@     pq@     �n@     ps@     pv@     �x@     �x@     �~@     ��@     �@     ��@     P�@     ��@     �@     0�@     L�@     ��@     ��@     ��@     4�@     �@     h�@     X�@     V�@     ��@     �@     B�@     .�@     O�@     ��@     v�@     v�@     ��@    ���@     ��@     <�@    `��@    X�!A     ��@     m�@    �K�@     ��@     ��@     2�@     ��@    ���@    �"�@     �@     %�@    ���@    �C�@     ��@    ���@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     �@     `{@     @\@      4@      @        
%
Conv2/weights/summaries/mean_18P��
'
 Conv2/weights/summaries/stddev_1� �=
$
Conv2/weights/summaries/max_1!T>
$
Conv2/weights/summaries/min_1�vU�
�
!Conv2/weights/summaries/histogram*�	   @ٮʿ   � ��?      �@! �ԏ��J�)g��;�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s�����Zr[v��I��P=��8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ�u��gr�>�MZ��K�>�f����>��(���>��Zr[v�>O�ʗ��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              L@     `�@     8�@     �@     �@     p�@     0�@     ��@     @�@     ��@     p�@     ��@     ��@     �@     Ѝ@     ��@      �@     �@     H�@     ��@     ��@     @�@     �~@     p|@      x@     �x@     pw@     �s@     �s@     `p@      p@     @k@      j@     �e@      d@     �`@      a@      \@      \@      Y@     �W@     �Y@     �S@      S@     @Q@      I@      E@      G@     �@@      F@      J@      <@      :@      =@      <@      >@      4@      7@      6@      *@       @      1@       @      @       @      &@      @      @      @      @      @      @      @      @       @      @      @      �?       @      @      �?       @      @       @      �?      �?      �?      @      @              �?               @              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?       @      �?              �?      @      @       @       @       @       @      @       @      @      @      @      @      @      @       @      "@      "@      ,@      (@      $@      *@      3@      0@      0@      0@      2@      @@      7@      B@     �D@      @@      F@     �D@      H@     �G@      L@     �L@      T@     �S@     �M@     @Q@     �V@     �\@     ``@     @^@     �a@      a@     @f@     `g@     �i@     @o@      m@     �n@     �r@     �r@     �u@     �w@     �z@     �|@     ��@     �@     ��@     0�@     ��@     �@     0�@     P�@     p�@     �@     ��@     В@     <�@     ��@     x�@     �@     p�@     x�@     `�@     ��@     ��@     (�@     �@@        
$
Conv2/biases/summaries/mean_1���=
&
Conv2/biases/summaries/stddev_1C�I;
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1�=�=
�
 Conv2/biases/summaries/histogram*q	    ��?   �پ�?      P@!   E�0@)�b����?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:              �L@      @        
�#
Conv2/conv2_wx_b*�"	    h�   ���?     $#A! P�m��@)�#JǪG�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(�['�?�;;�"�qʾG&�$��5�"�g������m!#�>�4[_>��>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              �?       @      @      =@     �[@     �n@     �~@     �@     ��@     �@     P�@     b�@     ױ@     �@     q�@     <�@     Q�@     O�@    ���@    �a�@    �l�@    �i�@    �n�@     �@     ��@    ��@     ��@     ��@    ���@    �B�@    �u�@     �@     ��@     ��@     ��@     4�@     [�@     ��@     ��@     =�@     a�@     �@     ��@     ��@     ��@     v�@     p�@     ƥ@     �@     �@     4�@     Ԙ@     4�@     T�@     �@     (�@     ȋ@     (�@     x�@     І@     �@     8�@     �@     �@     @�@     x@     �u@     �|@     �s@     �s@     �p@      o@      l@     �f@     �d@     �b@      c@     �_@     �b@     �r@     @^@     �T@      T@     �P@      Y@      I@     �J@      J@     �C@     �B@     �E@      A@      6@      >@      K@      8@      6@     �G@      3@      0@      *@      $@      @      "@      $@       @      @       @      @      @      @      @      @      @      @       @      �?      @       @      �?       @      @      @       @      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @               @              �?      �?      @      �?              �?       @      @       @      @       @      �?       @      @      @      @      @      @      @      @       @      @      @      @       @      &@       @      @      ,@      5@      8@      1@      7@      2@     �@@      5@      B@      3@      C@      G@     �C@      I@      C@     �Q@     �E@      R@     @i@     �U@     @\@      W@     @S@     �Z@      Y@     �b@     `b@      w@     �f@     �y@     �l@      j@     �n@     �p@     �p@     �s@     `u@     �z@     �y@     �@     �@     p�@     �@     ��@     ܐ@     �@     0�@     ԑ@     ��@     l�@     Й@     ��@     ��@     ��@     ~�@     ,�@     �@     ҥ@     :�@     ��@     �@     �@     Բ@     j�@     �@     ��@     )�@     G�@     ��@    �Y�@     �@    ���@    �B�@     )�@     �@     X�@     !�@     4�@    �&�@     [�@     B�@     R�@     Q�@     �@     K�@    �4�@     �@     ��@     �@     7�@      �@     �@     ��@     �@     ��@     p{@     �d@     @S@      2@      @        
�
Conv2/h_conv2*�   ���?     $#A! �J2!g�@)�sd�j�@2�	        �-���q=���m!#�>�4[_>��>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�	            <�A              �?              �?              �?              �?      �?               @               @              �?      �?      @      �?              �?       @      @       @      @       @      �?       @      @      @      @      @      @      @      @       @      @      @      @       @      &@       @      @      ,@      5@      8@      1@      7@      2@     �@@      5@      B@      3@      C@      G@     �C@      I@      C@     �Q@     �E@      R@     @i@     �U@     @\@      W@     @S@     �Z@      Y@     �b@     `b@      w@     �f@     �y@     �l@      j@     �n@     �p@     �p@     �s@     `u@     �z@     �y@     �@     �@     p�@     �@     ��@     ܐ@     �@     0�@     ԑ@     ��@     l�@     Й@     ��@     ��@     ��@     ~�@     ,�@     �@     ҥ@     :�@     ��@     �@     �@     Բ@     j�@     �@     ��@     )�@     G�@     ��@    �Y�@     �@    ���@    �B�@     )�@     �@     X�@     !�@     4�@    �&�@     [�@     B�@     R�@     Q�@     �@     K�@    �4�@     �@     ��@     �@     7�@      �@     �@     ��@     �@     ��@     p{@     �d@     @S@      2@      @        

cross_entropy_scl�y�>

training_accuracyfff?�(���S      ��	�`�k���A�*ާ
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@����߿+ٰ�i����ϟ?�~De�P�޼�{�,�&��'�N��������?�l�i��?L�&|�D�ab���R��?�`��詿��i��&)����?��9��NFF������jXt
L����`���Y�8��G0.9�n��������J�300�(30���a��R�f�?IbH|�sl��?>��bٲ?����!�K4�����C� �]ZKp�L    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`p ��Ǆ*�����$*�+�����N<�֊�dcd����&?�!)[��E���$��C�����߿�q�1�����X� K>� ���*����������g��g```8��{�����#�˵���p��  ˠ"��ϙ)    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  	IDAT(�ݐ=KBa�*����CA��C�����DC��)m--Ads���ŢA���IA�oC7|����������������Bi��'Ǩ\<^�!潙ַU���Rd�햷$���B�!�z������F2��Lq? �Rw����@�d���+������	��<��Wi$��U��s�E9���,���,��ن���v��&	����/5}���\l��vflS&KU�q?tN�%�'L8h���5{�u��~ �%�GiC    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   IDAT(�͑1KBa�Ko�!IA�%��@8�FQ�Q�"�AR�I�T�m��bQC-����PP]0M�hЛ�^tk���}缜�L> ��T���.�o�'u���� Dy��=l*�x���3�n�W�O���d�{��O����}�K�n6��	��&��pOj���$�8��Z��[I�^jkp�*�}(��F�a]0p)m0 yl5!�w[t���4�Rvi��f��
� �Kגr�>v�h����	�b�ζ�ǹs��gO
�@�r�}Vj������z����    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`@�Q1��~쒂���z��v�`q�m+p�m���8n٭t�xL���Pxh�dK*�V�4�/��P�������H�w�?8f���2NY��&�9�6���`�S���a.S��~���%i��O%�ǂ��R�Vc8��.�?����e(�  _�.F�8}    IEND�B`�
%
Conv1/weights/summaries/mean_1��W�
'
 Conv1/weights/summaries/stddev_1[�=
$
Conv1/weights/summaries/max_1aFL>
$
Conv1/weights/summaries/min_1��R�
�
!Conv1/weights/summaries/histogram*�	    �Zʿ    ̈�?      �@! �V]o�)����@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��u�w74���82���(���>a�Ϭ(�>�u�w74?��%�V6?ܗ�SsW?��bB�SY?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�               @       @      *@      .@      0@      5@      :@      3@      0@      5@      6@      0@      $@      4@      (@      2@      .@      2@      4@      &@      &@      &@       @      @      @      @       @      @      @      �?      @      @       @       @       @      �?      �?      @      �?      �?       @      �?              �?      �?      @      @              �?              �?              �?              �?               @              �?      �?      �?       @      �?      �?      �?      @      �?      �?      �?               @      @      @              @      @      @      @      @      @      @      @       @      "@      @      ,@      @      @      (@      (@      *@      0@      2@      ,@      6@      5@      "@      .@      4@      $@      5@      1@      (@      2@      (@        
$
Conv1/biases/summaries/mean_1F�=
&
Conv1/biases/summaries/stddev_1�Dg;
#
Conv1/biases/summaries/max_1���=
#
Conv1/biases/summaries/min_1��=
�
 Conv1/biases/summaries/histogram*�	   ��·?   �ֺ?      @@!   �� 	@)e�����?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @      :@      @        
�"
Conv1/conv1_wx_b*�"	   �*t��   `���?     $3A! "���@)�N#�Go�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ�H5�8�t�BvŐ�r�豪}0ڰ>��n����>G&�$�>�*��ڽ>��~���>�XQ��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @      f@     P@     ��@     ̐@     ��@     ��@     ��@     ��@     إ@     |�@     ��@     ��@     �@     ��@     ܲ@     ��@     ;�@     �@     J�@     ��@     ��@     u�@     [�@     ��@     '�@     .�@     }�@     ��@     ��@     P�@     |�@      �@     ƨ@     :�@     �@     p�@     *�@     ��@     ��@     ��@     D�@     �@     ��@     ��@     ��@     ��@     ��@      �@     p�@      �@     X�@     0�@     X�@     h�@     P}@     �z@     �z@     y@      t@     s@     �p@     `o@      l@     �h@     �h@      d@      c@     @`@     @\@      ^@      Z@     �T@      V@     �S@     @T@     �N@      Q@     �I@     �L@     �E@      E@     �@@     �C@      4@      ;@      <@      6@      ,@      2@      1@      0@      1@      8@      *@       @      "@      @      (@       @      @      @      @      @      @      @      @      @      @              @      @      �?      �?      @              �?      @              �?      �?              �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?               @               @               @              �?      @      �?              �?              �?       @       @      @      @      @      @       @      @      �?       @      @      @      @      @      @      @      @       @      (@      "@      ,@      "@      .@      *@      .@      0@      8@      3@      =@      9@      @@      5@      =@     �B@     �B@      F@      H@      I@     �E@      M@     �R@      S@      R@      W@      X@     �[@     �_@     �_@     �`@     `a@     �d@     `k@     �g@     �p@     �n@     �q@     �t@     �v@     �w@     0{@     �~@     ��@     ��@     ��@     X�@     ��@     (�@     ��@     \�@     h�@     @�@      �@     X�@     p�@     �@     z�@     ��@     �@     R�@     ��@     d�@     e�@     ߴ@     6�@     m�@    ���@    ���@    �&�@     O�@    ��"A     ��@     m�@    ���@     �@    ���@    ���@     ��@    �s�@     �@     f�@     \�@     1�@     @�@     %�@    �P�@     k�@     ��@     �@     �@     �@     d�@     ��@     Ȍ@     �y@      ]@      2@      @        
�
Conv1/h_conv1*�   `���?     $3A! H�2&=A)2�-t�@2�        �-���q=豪}0ڰ>��n����>G&�$�>�*��ڽ>��~���>�XQ��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            87A              �?              �?              �?               @               @               @              �?      @      �?              �?              �?       @       @      @      @      @      @       @      @      �?       @      @      @      @      @      @      @      @       @      (@      "@      ,@      "@      .@      *@      .@      0@      8@      3@      =@      9@      @@      5@      =@     �B@     �B@      F@      H@      I@     �E@      M@     �R@      S@      R@      W@      X@     �[@     �_@     �_@     �`@     `a@     �d@     `k@     �g@     �p@     �n@     �q@     �t@     �v@     �w@     0{@     �~@     ��@     ��@     ��@     X�@     ��@     (�@     ��@     \�@     h�@     @�@      �@     X�@     p�@     �@     z�@     ��@     �@     R�@     ��@     d�@     e�@     ߴ@     6�@     m�@    ���@    ���@    �&�@     O�@    ��"A     ��@     m�@    ���@     �@    ���@    ���@     ��@    �s�@     �@     f�@     \�@     1�@     @�@     %�@    �P�@     k�@     ��@     �@     �@     �@     d�@     ��@     Ȍ@     �y@      ]@      2@      @        
%
Conv2/weights/summaries/mean_1����
'
 Conv2/weights/summaries/stddev_1(�=
$
Conv2/weights/summaries/max_1�>T>
$
Conv2/weights/summaries/min_1E�V�
�
!Conv2/weights/summaries/histogram*�	   �h�ʿ   �݇�?      �@!@�n$�{K�)2:�|�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾��~]�[Ӿjqs&\�ѾR%�����>�u��gr�>K+�E���>jqs&\��>E��a�W�>�ѩ�-�>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �L@     8�@     H�@     h�@     ,�@     ��@     ��@     ��@     �@     ��@     ��@     ̒@     ��@     �@      �@     ��@     ��@     ��@     ��@     @�@     (�@      �@     �}@     �|@      x@     y@     pv@     `s@     �r@      r@     @o@     @k@     �h@     �d@     `d@     @a@     @b@     @]@     @\@     @X@     �U@      [@     �P@     @T@      T@      H@      O@     �G@     �B@     �C@      E@      =@      =@      <@      7@      4@      :@      3@      6@      &@      ,@      $@      "@      .@       @      *@      @      @      @      @      @      "@      @      @      �?      @      @      @              �?      �?      �?       @              �?       @      �?       @               @      �?              �?      �?               @              �?              �?              �?              �?              �?               @              �?              �?               @               @              �?      �?      @      �?      �?      �?       @      �?              @       @      @       @       @       @      @       @      @      @      @      @      @      $@       @      @      @      @      @      &@      $@      $@      ,@      ,@      .@      >@      B@      ?@      =@     �B@      ?@     �B@      D@     �H@      I@      P@      O@     �P@     �R@      U@     @P@      T@     @Z@     �]@     ``@     �a@     `c@     �f@     �f@      i@     �o@      l@      o@     �r@     @t@     �t@     �w@      {@     �}@     �@     �~@     8�@     0�@     h�@     8�@     @�@     H�@     (�@     ��@     ��@     Ē@     ,�@     ؒ@     h�@      �@     ��@     ��@     �@     ��@     �@     �@      A@        
$
Conv2/biases/summaries/mean_1]�=
&
Conv2/biases/summaries/stddev_1tP;
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1���=
�
 Conv2/biases/summaries/histogram*q	    2ط?    ���?      P@!   P�+@)z-�W���?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               M@      @        
�#
Conv2/conv2_wx_b*�#	   @��   �i�?     $#A! `ZS	�@)�b;�*�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��������?�ګ�u��6
��K���7����|�~�>���]���>�XQ��>�����>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�               @              @     �D@     @X@      m@     Pz@     @�@     ��@     \�@     (�@     L�@     ��@     ��@     c�@     �@     -�@     ؾ@     +�@    �r�@    �@�@    ���@     <�@    ���@     ��@     �@    �,�@    ��@    ���@     �@    ���@     k�@    �I�@     ��@     ��@     w�@     N�@     �@     ��@     ��@     ��@     ��@     �@     Z�@     d�@     �@     ��@     ��@     v�@     ��@     ��@     ��@     ؚ@     ��@     h�@     ��@     x�@     @�@     ��@     ��@     ��@     ��@     ,�@     P�@     �|@     X�@     `x@     �u@     �t@     �@     �t@      k@     pq@     �w@      j@     �[@     �[@     �b@     @\@     �U@     @U@     �R@     @R@      T@      Y@      ^@     �V@      H@     �@@      9@      ?@      L@      =@      6@      1@      ,@      8@      0@      ,@      .@      "@      @      @      @      0@      &@      @      @      $@       @      @       @      @      @       @      �?      @       @       @              �?      �?      �?      @       @       @      �?              @               @               @      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              <@              �?              �?      �?      �?      @      �?      �?      @      �?       @       @      @      @               @      @      @      @      C@      @      @      "@      "@      0@      *@      (@      3@     �J@      6@      ,@      ,@      9@      5@      ;@      2@      A@     �D@      J@      F@     �U@     �S@     �M@     �Q@     �Y@     `e@      V@     �V@      a@      `@     @_@      f@     �h@      o@     �j@     `n@      t@     �w@     w@     ��@     �|@     `z@     Ѓ@     ��@     ��@     X�@     �@     ��@     ��@     ��@     ��@      �@     h�@     h�@     h�@     :�@     d�@     Σ@     "�@     p�@     ̧@     �@     P�@     ��@     /�@     ��@     U�@     �@     Ÿ@     ��@     (�@    ���@     ��@    ���@     V�@     ��@     �@    ���@     ��@     n�@     ��@    � �@    �8�@    �y�@     &�@    ���@     ��@     ��@     ��@    ���@     2�@     	�@     �@     v�@     ^�@     ��@     ��@     P�@     @v@      c@     �O@      *@      @        
�
Conv2/h_conv2*�   �i�?     $#A! `��a�@)�Mg����@2�        �-���q=��|�~�>���]���>�XQ��>�����>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�            L�A              �?              �?              �?              �?      �?      �?              �?              <@              �?              �?      �?      �?      @      �?      �?      @      �?       @       @      @      @               @      @      @      @      C@      @      @      "@      "@      0@      *@      (@      3@     �J@      6@      ,@      ,@      9@      5@      ;@      2@      A@     �D@      J@      F@     �U@     �S@     �M@     �Q@     �Y@     `e@      V@     �V@      a@      `@     @_@      f@     �h@      o@     �j@     `n@      t@     �w@     w@     ��@     �|@     `z@     Ѓ@     ��@     ��@     X�@     �@     ��@     ��@     ��@     ��@      �@     h�@     h�@     h�@     :�@     d�@     Σ@     "�@     p�@     ̧@     �@     P�@     ��@     /�@     ��@     U�@     �@     Ÿ@     ��@     (�@    ���@     ��@    ���@     V�@     ��@     �@    ���@     ��@     n�@     ��@    � �@    �8�@    �y�@     &�@    ���@     ��@     ��@     ��@    ���@     2�@     	�@     �@     v�@     ^�@     ��@     ��@     P�@     @v@      c@     �O@      *@      @        

cross_entropy_sclq�M>

training_accuracy�k?���m�T      �#C�	t�n���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`H ��߿e�X%�>������;�B�p��[���Xd�l��ǢW���S�����������+��ef0��c``�ލ�����߿��~<>Ӹ����X�j�ծĿ���r�ν��Mq�2(���������gs�Ի����8��V<�;��-��vä_�al&��/&������p�?�&L�
�$��H�����A��ݰ���������߿������g``(��p�y�W�.�=  ��Z�C�W    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`H �w�{���q�����߿�Vs`�x��덯��5ĦU�L�����8��~���Vn�r��{�]N��߿[u�E�LK>��|8l�����w�8dT���w�
�,����.��e������ܓ����S�g�߿z��1�����E�c;��T�ú�� b�0I�m��~\;vV-F�Q��N��������߿���İ�����4cTT���WW���� UXcF�y    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   IDAT(�Ց/KA�߫^X�.)�,�咠�+6�=��	����� "�E4n8�E1[d�E��L��X��y����{��wfD�SjA�Ǳ���;�o~�;�P@�륺u��p�$}�#�/HDD2^[�[��\db6�
���Q��o��1����@��51�U�~�sdd�~Օ�q��W'��^����r5s�*+U��}R4���h<A:'F�`�AI��t;�g��:U}���
 ���)���ٍ��DF����m�u{�{�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�?`D�q�����+�O�FW�|�￿������MKdIE� ����x�u����τ*}�$#�:nɵ������4;��9(^y������ǵX������w�eF�<�"�����˨z�Wh�Hq�|�k����E���_����mP$����"��߿VT��������]y�?�����H�Ќ�a��ٿPuZ~���﷿�_�G�C�ο���3Ĕ�7  ��`��!�    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�?`Fbk��vE�	�-���)"��
� ��^��[RG����!Y�����]�_��y����Rl�u��񍁁���d
�&�20000�;���� ���]���S�n��)}������ ���yt���������a��6�1W��EC��o����o��A�$<�
���_����(��`(�?�o(9l�@�K,�x$g�����n  Ŷ?^���4    IEND�B`�
%
Conv1/weights/summaries/mean_1�OU�
'
 Conv1/weights/summaries/stddev_1�'�=
$
Conv1/weights/summaries/max_1��L>
$
Conv1/weights/summaries/min_1O�S�
�
!Conv1/weights/summaries/histogram*�	   �	}ʿ   `Y��?      �@! ������)X��}��@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��lDZrS�nK���LQ�k�1^�sO�IcD���L�+A�F�&�U�4@@�$�U�4@@�$?+A�F�&?nK���LQ?�lDZrS?E��{��^?�l�P�`?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�               @       @      *@      .@      0@      5@      ;@      1@      2@      6@      4@      1@      $@      2@      *@      3@      0@      (@      7@      *@      "@      &@      $@      @      @       @       @      @      @      @      @      @      @       @      �?       @      �?      @       @              �?              �?       @              �?       @       @      �?      �?              �?              �?              �?              �?              �?               @      �?      �?              @               @      @      �?               @       @      @       @      �?      @      @      @      @      @      @       @      "@      "@       @      @      *@       @      @      .@      $@      ,@      ,@      3@      *@      7@      5@      $@      0@      3@      "@      6@      1@      (@      2@      &@      �?        
$
Conv1/biases/summaries/mean_1?��=
&
Conv1/biases/summaries/stddev_1ޑl;
#
Conv1/biases/summaries/max_1���=
#
Conv1/biases/summaries/min_14�=
�
 Conv1/biases/summaries/histogram*�	    �Ʒ?    [ݺ?      @@!   ڧ�@)t�D|��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      9@      @        
�#
Conv1/conv1_wx_b*�"	   �%4��   @Ԭ�?     $3A! �u\%��@)w#SoZ�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|KվK+�E��Ͼ['�?�;;�"�qʾ�XQ�þ��~��¾0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @     �g@     (�@     �@     ��@     �@     8�@     ��@     0�@     D�@     
�@     ��@     \�@     ��@     n�@     e�@     ��@     ��@     �@     
�@     )�@     S�@     D�@     ��@     _�@     ~�@     �@     ��@     C�@     7�@     d�@     ,�@     t�@     ¨@     ��@     `�@     L�@     �@      �@     �@     ț@     ��@     ��@     H�@     ܔ@     ��@     ��@     ��@     Ѝ@     ȉ@     ��@     ��@     ��@     ؂@     �@     P~@     |@     Pz@     @v@     0w@     �r@     �r@     Pp@     @l@     �i@     �i@     �f@     �e@      b@      b@     @]@     �Y@     @X@      Y@     �S@      R@     �U@     �O@     �J@     �O@      I@     �A@      G@     �H@      :@      :@      4@      >@      5@      6@      1@      5@      ,@       @      *@       @      (@      @              @       @      @      "@       @       @       @      @      �?      @       @      @       @      �?      @       @       @       @      �?       @       @       @              �?              �?      �?              �?       @              �?      �?              �?              �?              �?              �?              �?       @      �?              �?      �?              �?               @      �?      @      �?       @      �?      @      �?      �?       @      �?      �?      @      @      @      @       @      @      @      @       @      @       @              @      $@       @      @      *@      $@      &@      "@      *@      4@      6@      0@      8@      3@      ;@      =@      D@      C@      C@     �I@      L@      P@      M@     �Q@     @R@     �R@      ]@     �Y@     @^@      _@     �_@     �_@     �c@     �f@      h@     �i@     �m@     0p@     �r@      t@     v@     Pv@     �x@     �}@     Ȁ@     ؂@     ȃ@     �@     p�@     p�@     Ў@     ��@     \�@     ��@     ��@     0�@     4�@     �@     4�@     ֣@     8�@     b�@     ��@     ��@     #�@     ҵ@     Y�@     �@     ��@    ���@    �2�@    �r�@    d�!A    Pf�@    ���@     V�@    ��@    ��@     ��@     *�@    ��@     ��@    ���@     ��@    ��@     \�@     �@     ��@     ��@     ĸ@     ^�@     ��@     F�@     ��@     H�@     �@     �~@      a@      C@      &@        
�
Conv1/h_conv1*�   @Ԭ�?     $3A! )e�_A)*���h�@2�        �-���q=0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            �A              �?              �?              �?              �?       @      �?              �?      �?              �?               @      �?      @      �?       @      �?      @      �?      �?       @      �?      �?      @      @      @      @       @      @      @      @       @      @       @              @      $@       @      @      *@      $@      &@      "@      *@      4@      6@      0@      8@      3@      ;@      =@      D@      C@      C@     �I@      L@      P@      M@     �Q@     @R@     �R@      ]@     �Y@     @^@      _@     �_@     �_@     �c@     �f@      h@     �i@     �m@     0p@     �r@      t@     v@     Pv@     �x@     �}@     Ȁ@     ؂@     ȃ@     �@     p�@     p�@     Ў@     ��@     \�@     ��@     ��@     0�@     4�@     �@     4�@     ֣@     8�@     b�@     ��@     ��@     #�@     ҵ@     Y�@     �@     ��@    ���@    �2�@    �r�@    d�!A    Pf�@    ���@     V�@    ��@    ��@     ��@     *�@    ��@     ��@    ���@     ��@    ��@     \�@     �@     ��@     ��@     ĸ@     ^�@     ��@     F�@     ��@     H�@     �@     �~@      a@      C@      &@        
%
Conv2/weights/summaries/mean_1�\��
'
 Conv2/weights/summaries/stddev_14	�=
$
Conv2/weights/summaries/max_1T>
$
Conv2/weights/summaries/min_1M�V�
�
!Conv2/weights/summaries/histogram*�	   �i�ʿ   �B��?      �@! =�|%�K�)��o��x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[���FF�G ���Zr[v��I��P=��pz�w�7��})�l a�ѩ�-߾E��a�Wܾ�iD*L�پ��~]�[Ӿjqs&\�Ѿ�XQ�þ��~��¾��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              N@     X�@     h�@     (�@     Џ@     ��@     (�@     ��@     �@     ��@     �@     ��@     ��@     �@     ȍ@     ȍ@      �@     (�@      �@     �@     X�@     ��@     @     p{@     �x@     @y@     �u@     Pt@     �r@     Pq@     �n@     �l@     @h@      f@      e@     �b@     @^@     @_@     @Z@     �Y@     �U@     �U@      T@     �R@     @S@     �P@     �F@      E@     �H@     �H@     �A@      >@      ;@      8@      8@      >@      .@      0@      (@      *@      *@      *@      @      &@      $@      "@      @      @      @      @      @      @      @      @      @      @      @      @       @      @       @      @               @      @      �?      �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              @      �?       @              �?              �?       @      �?       @      @      @      @      �?              �?      @       @              @       @       @      @       @      @      �?      @      @      "@      @      @      @      "@      $@      *@      @      $@      (@      .@      4@      4@      0@      8@      <@      9@      3@      9@     �E@      B@     �L@     �C@     �K@     @R@      K@     �P@      S@     �P@     �Q@     �R@     @[@      `@     @^@     �`@     @e@     �e@     �g@     �j@     �m@      n@     �o@     `q@     pt@     pu@     pw@     pz@     �~@     �@      @     �@     `�@     ��@     8�@     ��@     ��@      �@     (�@     Ȑ@     l�@     d�@     ̒@     d�@     �@     Ȓ@     p�@      �@     x�@     @�@     �@      A@        
$
Conv2/biases/summaries/mean_1D�=
&
Conv2/biases/summaries/stddev_1S�O;
#
Conv2/biases/summaries/max_1"n�=
#
Conv2/biases/summaries/min_1��=
�
 Conv2/biases/summaries/histogram*q	   ��ܷ?   @ĭ�?      P@!  ���(@)P�=����?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               M@      @        
�#
Conv2/conv2_wx_b*�#	   @����   ��?�?     $#A! �q$���@)�A�h���@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿjqs&\�Ѿ0�6�/n���u`P+d��X$�z�>.��fc��>���?�ګ>����>5�"�g��>G&�$�>�XQ��>�����>K+�E���>jqs&\��>��~]�[�>��>M|K�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              �?      @      I@     @^@      q@     p}@     h�@     <�@     4�@     �@     �@     L�@     8�@     V�@     ��@     [�@    ��@     t�@    �V�@    ���@     ��@    ��@    �]�@     d�@     ��@    �+�@    ��@     +�@     ��@     {�@    ���@     Z�@     �@     ;�@     ��@     �@     ��@     �@     g�@     K�@     �@     ��@     �@     ��@     �@     ��@     �@     ^�@     �@      �@     ě@     \�@     ��@      �@     Г@     `�@     ��@     x�@      �@     �@     P�@     Ё@     �|@      �@     �~@     0x@     �v@     �y@     �z@     @y@     �h@     0q@     �n@     �o@     �b@     �`@     �`@     �e@     @[@     �W@      ^@     �R@     @R@     �`@     �F@     �E@      E@      E@     �B@     �C@     @T@      :@      4@     @R@      1@      4@      :@      .@      &@      *@     �g@      ,@      "@      ,@      @      @      @      @      �?       @      "@      "@              @       @      @       @      �?       @       @      �?              @       @              @              �?       @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?      @              �?       @      @      �?      @      @      @      @      @      @      @       @      @      "@      @      @      @      @      ,@      $@      .@      &@      (@      &@      .@      C@     �L@      3@     �F@      <@      6@      <@      G@      >@      J@     �H@     �a@      K@      L@     �M@     �O@      S@     �\@      c@     �X@     `g@     ``@     `b@     �a@     `e@     �i@     �k@     P�@     �o@     p|@     �@     0r@     �~@     �y@     ��@     H�@     �@     ��@     ��@     �@     ��@     ,�@     ��@     ��@     l�@     |�@     0�@     t�@     �@     V�@     ƣ@     ��@     &�@     ��@     ��@     ��@     ��@     -�@     �@     f�@     �@     G�@     ��@     ��@    ��@     ��@    ��@    �X�@     N�@    �<�@     C�@     ��@     �@    ��@     ��@    ��@     �@     ��@    ���@     P�@     �@     ��@     �@     N�@     ۳@     v�@     Ч@     Ƞ@     ��@     (�@     �y@     `i@     @R@      .@       @        
�
Conv2/h_conv2*�   ��?�?     $#A! �kq��@)O�=	�@2�        �-���q=X$�z�>.��fc��>���?�ګ>����>5�"�g��>G&�$�>�XQ��>�����>K+�E���>jqs&\��>��~]�[�>��>M|K�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�            8�A              �?              �?              �?              �?              �?               @              �?              �?      @              �?       @      @      �?      @      @      @      @      @      @      @       @      @      "@      @      @      @      @      ,@      $@      .@      &@      (@      &@      .@      C@     �L@      3@     �F@      <@      6@      <@      G@      >@      J@     �H@     �a@      K@      L@     �M@     �O@      S@     �\@      c@     �X@     `g@     ``@     `b@     �a@     `e@     �i@     �k@     P�@     �o@     p|@     �@     0r@     �~@     �y@     ��@     H�@     �@     ��@     ��@     �@     ��@     ,�@     ��@     ��@     l�@     |�@     0�@     t�@     �@     V�@     ƣ@     ��@     &�@     ��@     ��@     ��@     ��@     -�@     �@     f�@     �@     G�@     ��@     ��@    ��@     ��@    ��@    �X�@     N�@    �<�@     C�@     ��@     �@    ��@     ��@    ��@     �@     ��@    ���@     P�@     �@     ��@     �@     N�@     ۳@     v�@     Ч@     Ƞ@     ��@     (�@     �y@     `i@     @R@      .@       @        

cross_entropy_scl�3>

training_accuracy��u?8��cT      ���	��Mp���A�*Ө
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H  IDAT(���!H��ϛ���4\Ѽ"
&W��dq`Y��	��a�h6�C�O�1�@q����<���*~����{�?�n����2��v�����\�M�����њ�$~���Z\R�����% ؟��lm�h�Iuܘ�
p����W����$=������ �K��s�Şq�?-v�?ƫ�j�yLT��0x^V��O��ݵ����G�z�{67�q���֔j/�V([>�k�a�T����?�'������>t    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`Ā+p鯓}rX��?�{y��o�X�$?,teg`o�ڀ)�d'��c0!&�䗵�~�3Ő|q�0fg��f�;�p�	���C�3���<r�W�ݎ�e���<��,(�v,;Epid``Xu�Ä.+�O�zy����8�dJ����F`d``��ϖ��[y��SW����,{���<IR�G>  �:9Y�Č    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  IDAT(�c`p�����'�<X��������߿3�1%�����>�r�/�73000� IV�000\�� ��i�����d``��щ����7¼�.Y���pF�gh���������b�I�1b`8a�i��������Y��߿?��$����e�>���Y疿�BS����an'Éw,%ە�!k�?���Tv5������Vφ,���߿Ӛ?A��
�c�� ��{�4�$��dB�B���10000��c����PZ��?�!���# ���@�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�u��E��\S>�h=���߿�1��������߿w�10000"ə�8��f`8���>�SOX�p����}P��ᬼ�h!�����4:��������c����5���l���a1���K�&��p�}>	IE1�-�B�^�H�qV��$�W1��7������\豫s��%��38CsF��O������SK�R��J����Ð  ��L����    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`� �nɕm�l&�'#��ǔ��Z�������A�8�|�����RŰ�;�����n��3���?(��M�����200p���UN��߿����$^����dTɌL�����������Q%��Xd�W}���/�$��-����o:n��s@�L�w�qJ�cx���S)f��09�w�6���\j  �_D��B�    IEND�B`�
%
Conv1/weights/summaries/mean_1O,P�
'
 Conv1/weights/summaries/stddev_1�F�=
$
Conv1/weights/summaries/max_1ۄL>
$
Conv1/weights/summaries/min_1�,T�
�
!Conv1/weights/summaries/histogram*�	   ���ʿ   `���?      �@!  �kTT�)��S�ݨ@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� ��o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�IcD���L��qU���I���[�?1��a˲?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?E��{��^?�l�P�`?���%��b?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      @      *@      .@      0@      5@      <@      0@      3@      3@      3@      4@      $@      3@      (@      2@      0@      .@      3@      ,@      "@      (@       @      @      @      @      "@      @      @      @      @       @      @              @              @       @       @      �?              @              �?      �?              �?              @              �?              �?              �?              �?              �?              �?       @              @      @       @       @      �?              �?      �?      @      @       @      @      @      @       @      @      @      @       @      "@       @       @      (@      "@      @      (@      &@      0@      (@      4@      ,@      6@      5@      $@      .@      4@      "@      6@      1@      (@      1@      (@      �?        
$
Conv1/biases/summaries/mean_1���=
&
Conv1/biases/summaries/stddev_1&�t;
#
Conv1/biases/summaries/max_1���=
#
Conv1/biases/summaries/min_1	.�=
�
 Conv1/biases/summaries/histogram*�	    ���?    ���?      @@!   |��@)�4K\��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @      :@      @        
�"
Conv1/conv1_wx_b*�"	   �}���   @?��?     $3A! z��5��@)	ޝSO��@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;�*��ڽ�G&�$����������?�ګ����?�ګ>����>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>�_�T�l�>�iD*L��>E��a�W�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @     �\@     �v@     ��@     h�@     ��@     ��@     ��@     B�@     h�@     
�@     ��@     `�@     ��@     i�@     ��@     �@     ��@     ��@     �@     ѵ@     E�@     6�@     �@     ��@     >�@     A�@     ��@     D�@     ذ@     ֯@     F�@     �@     T�@     f�@     ~�@     �@     ��@     ��@     `�@     ؜@     �@     ܘ@     H�@     D�@     `�@     d�@     h�@     �@     ��@     @�@     ��@     0�@     ��@     ؂@     �}@     �~@     �z@     0x@     �t@     Ps@     `r@     �p@      p@     `n@      k@      c@     �e@     @b@     �`@      ^@     @W@     �Y@     �U@     �Z@     �V@     �T@      P@     �P@      Q@     �A@     �F@      B@      D@      B@      >@      5@      7@      4@      7@      ,@      6@      $@      :@      @      @      (@      @      @      "@      @      @       @       @      @      @      @      @              @              @      @      �?              �?      @      @       @      �?               @      �?      �?              �?              �?      �?              �?              �?              �?               @              �?              �?       @              �?      �?              �?      �?       @      �?      @       @       @      �?      �?              @       @      �?      @              @      @      @      @      &@      @       @      @       @      &@      @      $@      $@      *@      @      @      *@      3@      9@      7@      8@      5@      3@     �B@      E@     �B@     �I@     �F@      E@     �M@     �K@      K@      O@      U@     �V@     �Y@     �Z@     @[@     �^@     �c@     �c@     �e@      d@     �k@     `l@     �n@      r@     ps@      u@     pv@     z@     |@     ��@     ȁ@     (�@     �@     ��@     ��@     ��@     d�@     �@     ��@     ��@     l�@     x�@     $�@     p�@     ~�@     �@     ܦ@     ު@     ��@     4�@     ��@     k�@     ��@     v�@     ��@    �2�@     I�@    ��@    j�!A    `�@    �Q�@     d�@    ��@     ��@    �*�@     ��@    ���@    ���@     ��@    ���@     �@     �@     �@    �C�@     �@     ��@     T�@     ��@     ȩ@     ֠@      �@     �@     �z@     �]@      B@      @        
�
Conv1/h_conv1*�   @?��?     $3A! �i��WA)I4<Kk+�@2�        �-���q=���?�ګ>����>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>�_�T�l�>�iD*L��>E��a�W�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            �A               @              �?              �?       @              �?      �?              �?      �?       @      �?      @       @       @      �?      �?              @       @      �?      @              @      @      @      @      &@      @       @      @       @      &@      @      $@      $@      *@      @      @      *@      3@      9@      7@      8@      5@      3@     �B@      E@     �B@     �I@     �F@      E@     �M@     �K@      K@      O@      U@     �V@     �Y@     �Z@     @[@     �^@     �c@     �c@     �e@      d@     �k@     `l@     �n@      r@     ps@      u@     pv@     z@     |@     ��@     ȁ@     (�@     �@     ��@     ��@     ��@     d�@     �@     ��@     ��@     l�@     x�@     $�@     p�@     ~�@     �@     ܦ@     ު@     ��@     4�@     ��@     k�@     ��@     v�@     ��@    �2�@     I�@    ��@    j�!A    `�@    �Q�@     d�@    ��@     ��@    �*�@     ��@    ���@    ���@     ��@    ���@     �@     �@     �@    �C�@     �@     ��@     T�@     ��@     ȩ@     ֠@      �@     �@     �z@     �]@      B@      @        
%
Conv2/weights/summaries/mean_1�k��
'
 Conv2/weights/summaries/stddev_1��=
$
Conv2/weights/summaries/max_1BlT>
$
Conv2/weights/summaries/min_1�XW�
�
!Conv2/weights/summaries/histogram*�	    �ʿ   @���?      �@!@mK��M�)�A,��x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�*��ڽ>�[�=�k�>��~���>�XQ��>���%�>�uE����>�f����>��(���>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �O@     8�@     `�@     X�@      �@     ��@     �@     |�@     ,�@     �@     �@     ��@     X�@     x�@     ��@     ��@     ��@      �@     H�@     h�@     x�@     @�@     p~@     P|@     �x@      y@     `v@     ps@     �q@     @q@     pq@     �j@     �f@     `h@     `c@     �c@      `@     �\@      Z@      Y@     @X@     �T@     @U@     �Q@     �P@     �Q@     �M@      D@      J@     �E@      B@     �@@      >@      8@      ;@      .@      2@      0@      &@      7@      *@      $@      @      &@      &@      @      (@      $@      @      @      @      @      @      @      @      @       @      @       @      @      �?      �?      @       @       @              �?              �?       @              �?       @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      @              �?      �?       @      �?       @              �?      @              @      @      @      �?      @      @      @      @      @      @      @      $@       @      &@      "@      @      "@       @      0@      0@      ,@      4@      6@      3@      5@      4@      ?@      B@     �A@     �C@     �@@     �J@     �J@      Q@      M@      R@     �P@      O@     �S@     �X@      W@     �]@     �^@     �b@      d@     `f@     �h@      h@     �m@     �m@     Pp@     �r@     �s@     0u@     �x@     �y@     0~@     @     �@     �@     ؄@     Ѕ@     ؉@     ��@     P�@     ��@     ��@     ��@     ��@     (�@     Ԓ@     T�@     $�@     В@     x�@     �@     ��@     �@      �@      ?@        
$
Conv2/biases/summaries/mean_1� �=
&
Conv2/biases/summaries/stddev_1��O;
#
Conv2/biases/summaries/max_1Y��=
#
Conv2/biases/summaries/min_1a�=
�
 Conv2/biases/summaries/histogram*�	   �"̷?    ��?      P@!   � @)�cD����?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?      M@      @        
�#
Conv2/conv2_wx_b*�#	   ��J��   �y��?     $#A! ����@)�"�yu�@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澙ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;��������?�ګ���|�~���MZ��K��w`f���n>ہkVl�p>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>�ѩ�-�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              @      F@     �]@     �o@     p~@     �@     t�@     ��@     �@     ڬ@     {�@     I�@     �@     +�@     �@    ��@    �T�@     p�@    ��@    �D�@    ���@     �@     2�@     u�@    ���@     E�@     J�@    � �@    ���@     R�@    ���@     ��@     ��@     U�@     F�@     t�@     W�@     
�@     �@     �@     �@     �@     ��@     T�@     ��@     �@     �@     ȝ@     ~�@     8�@     ��@     ��@     `�@     ��@     p�@     ��@     `�@     X�@     ��@      �@     `�@     ��@     �@     �z@     ��@     �x@     0s@     �p@     �n@     �k@     �o@     �i@     �p@      e@     �`@     @a@     �[@      \@     �n@      V@      d@      L@     @Q@     �O@      G@     �N@     �H@      C@      C@     �I@     �P@      <@      6@      5@      4@      1@      1@      *@      0@      .@      $@      6@      $@      $@       @      @      @      @      @      @      @      @      @      @       @      @       @      @      �?              @       @       @       @              @      �?              �?              �?      �?      �?      �?              �?              �?              �?               @               @       @              �?      �?      �?              �?      9@      @      @      �?              �?              @      @      @      @       @       @       @      @      @      @      &@      $@      @      $@      $@      "@      $@      (@     �i@      .@      2@      2@      .@      6@      0@      :@      I@      A@      C@      D@     �G@      F@     �Q@      I@      V@     �U@     �Q@      W@     @\@      ]@     �f@     @i@      `@      b@     �n@      f@     `o@     q@     @m@     `k@     @|@     �r@     8�@     �}@     P~@     H�@     �|@      �@     `�@     Ћ@     X�@     x�@     ��@     �@     �@     Ȕ@     �@      �@     ��@     "�@     2�@     T�@     ��@     �@     ��@     �@     ޯ@      �@     y�@     w�@     ��@     ϻ@     ��@     ��@    �z�@    �!�@     a�@    �N�@     �@     R�@     6�@    ���@    ��@     ��@    ���@    �`�@    ���@     ,�@     2�@    ���@     ��@     ��@     ��@     ��@     ��@     B�@     ��@     ��@     ��@     ��@     `�@     �v@     @c@     �O@      7@       @       @        
�
Conv2/h_conv2*�   �y��?     $#A! v��@)sd����@2�        �-���q=w`f���n>ہkVl�p>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>�ѩ�-�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�             2A              �?               @               @       @              �?      �?      �?              �?      9@      @      @      �?              �?              @      @      @      @       @       @       @      @      @      @      &@      $@      @      $@      $@      "@      $@      (@     �i@      .@      2@      2@      .@      6@      0@      :@      I@      A@      C@      D@     �G@      F@     �Q@      I@      V@     �U@     �Q@      W@     @\@      ]@     �f@     @i@      `@      b@     �n@      f@     `o@     q@     @m@     `k@     @|@     �r@     8�@     �}@     P~@     H�@     �|@      �@     `�@     Ћ@     X�@     x�@     ��@     �@     �@     Ȕ@     �@      �@     ��@     "�@     2�@     T�@     ��@     �@     ��@     �@     ޯ@      �@     y�@     w�@     ��@     ϻ@     ��@     ��@    �z�@    �!�@     a�@    �N�@     �@     R�@     6�@    ���@    ��@     ��@    ���@    �`�@    ���@     ,�@     2�@    ���@     ��@     ��@     ��@     ��@     ��@     B�@     ��@     ��@     ��@     ��@     `�@     �v@     @c@     �O@      7@       @       @        

cross_entropy_scl�ڄ=

training_accuracyH�z?��>\�S      �aQ	1��r���A�	*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`d��⏕c(���p��+�A�E������ϟ?V��	w��{a{����?���I.�sIO���a��?J�m�Q�����}��L؜����
X�ͺ�ϟ?�԰�I�ϟ?�~��M��|'�+/��b�@``�g0�ax���l����g�d���������8%� �dճ��?�C���͟���
j����[�.�t��7�҄��܊@�+y��s��w��Y���=T d�_��4�I    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`Āw�?+&$6����.xLUx��/Nc�%>�b`��¦3��?K��T�H���W��A����BR���O[� ],:�����ǿ����"���߿��Er23Br�?ɫ�ߧ���^���n�sYP$�0�30<ª�N\rj�� xL����_/��d��Ȇ�������Թ�Ç#8u2$<3�-I%  ��B�9��x    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  
IDAT(�c`d���"S��,�ܗNM�����g���RB{���/3�P伯��b�����֞����d�
��9������{����=r^��700000�0000���@.9��9����7g```�^���$y�4�s^}Bq^���k��mB�4���?�Xkɷ���_��*P����G;��O����M�V0[n�z�" �ق�a�J�x��F�������Nc@�I��r�z�B�H�LIL!g֟��0H [H{4�G�H    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�	`F��o:�Si�{I��<���eB�	=�-)�ŀ[ҟ�$;�[ܒ{�H��G��<�;�8�����I~ܒ;��1��f3�����[R���V� �'�ߘ����������]	 ���.��    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H  	IDAT(�͑1KA����B�hB!؈��@�`!W�Xh�B$�Pk;�CD����	� D�l���2� �X���g����{0��J�L#)��K_���^-��\�1�{Cl��X���7�08�H� �(�p��V#��X�V���s�Н7��-@>�/��z��;����<[�7%�ta�]��@�rY���Ҽ���?�AC�TZjv |�A+��^LEg�I�GJ�ܟ"�W�������k��$�� �����hu�P��?�7�It���    IEND�B`�
%
Conv1/weights/summaries/mean_1�S�
'
 Conv1/weights/summaries/stddev_1dr�=
$
Conv1/weights/summaries/max_1;=L>
$
Conv1/weights/summaries/min_1t�T�
�
!Conv1/weights/summaries/histogram*�	   ���ʿ   `���?      �@!  6o���)Ӓc�K�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS��!�A����#@���%�V6��u�w74��lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @      @      *@      1@      ,@      4@      =@      2@      1@      3@      4@      3@      (@      2@      (@      1@      1@      ,@      3@      ,@      &@      "@      $@      @      @      @      $@      @      @      @      @      @              @       @      �?      �?       @       @      �?      �?       @      �?       @              �?      �?              �?      �?               @              �?               @              �?              �?              �?               @              @      @      �?       @              �?      �?      �?      @      @      @       @       @       @       @      @      @      @      "@       @      "@      "@      "@      $@      @      *@      $@      ,@      .@      1@      1@      4@      6@      $@      ,@      4@      $@      5@      3@      $@      2@      *@        
$
Conv1/biases/summaries/mean_1��=
&
Conv1/biases/summaries/stddev_1Q�{;
#
Conv1/biases/summaries/max_1d��=
#
Conv1/biases/summaries/min_1Iڻ=
�
 Conv1/biases/summaries/histogram*�	    I{�?   ���?      @@!   �a�@)p���%�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      9@      @        
�"
Conv1/conv1_wx_b*�"	    ���   @}��?     $3A! �����@)��,��@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE����jqs&\�ѾK+�E��Ͼ
�/eq
Ⱦ����žR%�����>�u��gr�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @     `e@      �@     �@     �@     ��@     ��@     F�@     �@     d�@     Ʃ@     �@     ư@     |�@     ղ@     ��@     L�@     ��@     )�@     N�@     I�@     v�@     ͵@     ݴ@     ��@     [�@     ��@     ��@     U�@     D�@     .�@     &�@     ��@     ҩ@     ��@     ڥ@     Σ@     L�@     ��@     B�@     L�@     �@     x�@     $�@     ��@     ��@     ��@     x�@     X�@     ��@     �@     ��@     Ȅ@     (�@     ��@     �}@      }@     Px@     pw@     �t@      t@     Ps@      o@     pp@      i@      l@     �e@     `a@     �b@     `a@      ]@     @Z@     �V@     �X@      V@     @P@      V@     �O@      H@     �K@      H@     �A@      D@      A@      :@      :@      7@      8@      8@      1@      ,@      1@      4@      ,@      2@      *@      @      @      @      (@       @      "@      �?      @      @      @      @      @      @      @      �?      @       @      �?       @      @      �?              �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @       @              �?              �?              �?      @      �?      �?       @      @      @       @      �?      �?      @      @      @       @      "@       @      �?      @      @      @      "@       @       @      ,@      $@      1@      $@      &@      &@      4@      1@      5@      (@      <@      <@      =@     �B@      B@      I@      O@      J@      M@      N@     �Q@      T@     @Y@     @Z@      W@     �]@     �_@     �e@      f@     �f@     �e@     �j@     �m@     �q@     �q@     `s@     u@     Px@     �{@     �@     �@     ��@     ��@     ��@     ؇@     Ȍ@     ��@     0�@     ��@     ܔ@     H�@     ��@     �@     H�@     ܡ@     �@     ̦@     @�@     ��@     ��@     ²@     ��@     ظ@     A�@     B�@     ��@    �T�@    ���@    .�!A    �a�@     U�@    ��@    ���@     X�@     r�@     I�@    ���@     ��@     ��@    ���@    �E�@    �_�@     B�@     j�@     2�@     i�@     �@     z�@     R�@     ��@     ��@     ؍@      {@     @b@      C@      @        
�
Conv1/h_conv1*�   @}��?     $3A! ��)pZA)2Q���W�@2�        �-���q=R%�����>�u��gr�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            ��A              �?              �?              �?              �?              �?               @               @       @              �?              �?              �?      @      �?      �?       @      @      @       @      �?      �?      @      @      @       @      "@       @      �?      @      @      @      "@       @       @      ,@      $@      1@      $@      &@      &@      4@      1@      5@      (@      <@      <@      =@     �B@      B@      I@      O@      J@      M@      N@     �Q@      T@     @Y@     @Z@      W@     �]@     �_@     �e@      f@     �f@     �e@     �j@     �m@     �q@     �q@     `s@     u@     Px@     �{@     �@     �@     ��@     ��@     ��@     ؇@     Ȍ@     ��@     0�@     ��@     ܔ@     H�@     ��@     �@     H�@     ܡ@     �@     ̦@     @�@     ��@     ��@     ²@     ��@     ظ@     A�@     B�@     ��@    �T�@    ���@    .�!A    �a�@     U�@    ��@    ���@     X�@     r�@     I�@    ���@     ��@     ��@    ���@    �E�@    �_�@     B�@     j�@     2�@     i�@     �@     z�@     R�@     ��@     ��@     ؍@      {@     @b@      C@      @        
%
Conv2/weights/summaries/mean_1�W��
'
 Conv2/weights/summaries/stddev_1��=
$
Conv2/weights/summaries/max_1@0T>
$
Conv2/weights/summaries/min_1�>W�
�
!Conv2/weights/summaries/histogram*�	   ���ʿ    ��?      �@!@��!�N�))�o�;�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��I��P=��pz�w�7���h���`�8K�ߝ�jqs&\�ѾK+�E��ϾX$�z�>.��fc��>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �P@      �@     ��@     ��@     4�@     ��@     ��@     ē@     �@     �@     <�@     Ԓ@     l�@     �@     ȍ@     ��@     ؉@     ��@     ��@     ��@     ��@     p�@     �~@     �|@     @x@     `y@     �v@     ps@     �q@     �p@     �p@     �i@     �i@     �d@     �e@     �d@     @`@      [@     �X@     @T@     �Y@     �X@     �S@     �R@      T@     �J@     �N@     �J@     �B@      C@      E@      ?@      7@      A@      9@      2@      .@      2@      3@      .@       @      0@      0@      0@       @       @      $@      @      @      @      @      @      @      @      @      @       @       @       @      @      @      @               @      @      �?      �?      @              @               @              �?              �?              �?              �?              �?               @      �?      �?              �?      �?              @              @      �?              �?      �?              �?       @      �?      �?       @       @      @      @       @      @       @       @      @      @       @      @       @      "@      ,@       @      2@      (@      $@      (@      0@      (@      8@      1@      8@      9@      =@     �A@      ;@     �D@      A@     �M@      G@     �K@     �M@     �Q@      I@     �T@     �W@      W@     @W@      Z@     �_@      b@     �c@      g@     �h@     �h@      n@     �l@      p@     @r@     t@     �u@     �x@     �z@     �}@     �~@     �@     �@     ��@     ��@     0�@     ��@     `�@     ��@     (�@     ��@     |�@     L�@     Ȓ@     h�@     �@     ��@     ��@     ��@     (�@     X�@     �@      A@        
$
Conv2/biases/summaries/mean_1��=
&
Conv2/biases/summaries/stddev_1�JO;
#
Conv2/biases/summaries/max_1���=
#
Conv2/biases/summaries/min_1=
�
 Conv2/biases/summaries/histogram*�	   �ѷ?    }��?      P@!  ���@)�!��P��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @     �L@      @        
�#
Conv2/conv2_wx_b*�#	   ��E�    }n�?     $#A! �6����@)��L|���@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�5�L�����]����:�AC)8g>ڿ�ɓ�i>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�               @      (@      I@     �d@     `u@     ��@     �@     Ԗ@     �@     �@     ��@     �@     e�@     ��@     Ի@     s�@    ���@    ���@     ��@     ��@     B�@    ���@     3�@     {�@     ��@    ���@    �_�@     ��@     ��@    �b�@     ��@    �O�@     %�@     <�@     ż@     P�@     5�@     ��@     ��@     �@     ��@     ��@     �@     ��@     �@     6�@     �@     R�@     ��@     ��@     �@      �@     �@     ؙ@     @�@     H�@     D�@     ��@     �@     ��@     ��@     ��@     ��@     @�@      y@     �u@     �|@     ps@     �r@     pr@     `n@     �j@     �r@     �h@     @m@     �_@     �^@     @^@     �[@     �V@      U@     �R@     �T@      P@      I@     �B@      N@      B@      ?@      @@     �F@     �V@      ;@      >@      ,@      0@      0@      .@      1@      &@      *@      (@      ,@      @      ;@      "@      "@      @       @      @      @       @      @      @      �?       @      @       @      5@      @      @       @       @       @              �?      @              �?              �?               @              �?              �?              �?              �?      �?      �?      �?              �?      �?      �?              �?      �?              �?       @      �?      �?              �?       @               @      @      @       @      @      @      @      @      $@      @      @      @      @       @      &@       @      "@      *@      &@      2@      *@      0@      4@      1@      @@      3@      A@      ;@      @@      ?@     �r@     �H@      F@     �M@     �N@     �S@     �V@     @T@     �U@     @Y@     �[@     �a@     `q@     �g@     @i@     �b@     `e@     `l@     �n@     0t@      t@     Ȁ@     �x@      |@     ��@     0z@     ��@      �@     ؂@     ��@     `�@     ��@     �@     ��@     x�@     X�@     ě@      �@     ��@     ��@     .�@     V�@     t�@     ب@     N�@     ��@     ��@     ְ@     !�@     m�@     ��@     o�@     ��@     ͼ@    ���@    ���@    ���@     ��@     \�@     "�@    ���@     �@    ���@    �p�@    ���@     �@     ��@     ��@    ���@    ���@    �a�@     _�@     K�@     ^�@     ̷@     ͳ@     �@     ��@     ��@     �@     �@     �w@      e@      J@      2@       @        
�
Conv2/h_conv2*�    }n�?     $#A! m$����@)��J!��@2�        �-���q=:�AC)8g>ڿ�ɓ�i>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�            �FA              �?              �?      �?      �?      �?              �?      �?      �?              �?      �?              �?       @      �?      �?              �?       @               @      @      @       @      @      @      @      @      $@      @      @      @      @       @      &@       @      "@      *@      &@      2@      *@      0@      4@      1@      @@      3@      A@      ;@      @@      ?@     �r@     �H@      F@     �M@     �N@     �S@     �V@     @T@     �U@     @Y@     �[@     �a@     `q@     �g@     @i@     �b@     `e@     `l@     �n@     0t@      t@     Ȁ@     �x@      |@     ��@     0z@     ��@      �@     ؂@     ��@     `�@     ��@     �@     ��@     x�@     X�@     ě@      �@     ��@     ��@     .�@     V�@     t�@     ب@     N�@     ��@     ��@     ְ@     !�@     m�@     ��@     o�@     ��@     ͼ@    ���@    ���@    ���@     ��@     \�@     "�@    ���@     �@    ���@    �p�@    ���@     �@     ��@     ��@    ���@    ���@    �a�@     _�@     K�@     ^�@     ̷@     ͳ@     �@     ��@     ��@     �@     �@     �w@      e@      J@      2@       @        

cross_entropy_scl��>

training_accuracyףp?N��f
U      ��9T	̙�t���A�
*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�����S���߿�㐛����ߏ����_������W��Ę`;�>34m����������T.l���w�r0���wpz�����>La��
�[��\$�=[���c!�z���߿����&Cn�￿!����������u?��[�nc``����߿a�ә#�M��Ê�߫*9Q#���:oB���̜A���K(�v�$��Q$�����������/�  )�l5�t�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@��?�l-�!i�/��;/CvT��߿W1$yW
	��z��l�0�zY�	M��.����������䬿�������	���"0ɿoڣ�9�|~]����L���̵�ѝ��W���t� ���U����.p��iT��/�섿%(r���~ͅ0m�;��"�V�����""s�?�{W
݉s��������q9���29� �,f�`�W    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  'IDAT(����/Ca��F�!� �HX|ej�H|E��J�$�A"�6�:6A��U$�.�.XJ��s\�{���Z{��9�9�}����p|��,~��R)��57��$�$�N���L��b��M����d��I�vqz6���������{��xQR�X ��K�޺ø�K����Dy����ai暙�f7Q �>d�.5Eօ��pŬJ: ��G����$�J�	�\�Y����KfJ�Y�G�q���?e���Xߏ��Wd��[2U_<��nаsQ|�F4(��'+�4�da���Ƕ�/�"�&��*    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�ő�JQ���)RX��mzK;M��-|`��P,W�N�7X�� haa!X��&��?���a-���޵�S����93��U��$)�E�� ��>E'�_��o8^�<��˝wV���|����J�+�v7��N �߯:p3S���7�N�o7~��fw(�Ro6�:�0��lvς�p��!�.F��O�#�r0�4���U�e�q���n�~r�^�����'�ۜ�P�����n���ð��n�O%Yu�P�c    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�?`F�����x��dܒ���"8L�r�"�qj��w���6��ũQ��?$.���jQ%���e�س�8������|qK0��� �'�$������C�>�-����ף�8�d���/.�=���K  ��"�b��    IEND�B`�
%
Conv1/weights/summaries/mean_1v�N�
'
 Conv1/weights/summaries/stddev_1���=
$
Conv1/weights/summaries/max_1a�K>
$
Conv1/weights/summaries/min_1�U�
�
!Conv1/weights/summaries/histogram*�	    �ʿ    ,w�?      �@!  �as.�)[*�H�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�d�\D�X=���%>��:���%�V6��u�w74�<DKc��T?ܗ�SsW?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @      @      *@      1@      ,@      7@      :@      1@      0@      4@      3@      5@      *@      .@      0@      .@      1@      *@      3@      .@      (@      $@      @       @      @      @       @       @       @       @       @      @              @      �?      �?      @      �?       @       @       @      �?      �?       @      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?       @      @      �?      @              �?              @       @      @      @       @      @      @      @      @      @      @       @       @       @      $@      "@      $@      @      *@      $@      *@      2@      .@      1@      4@      6@      $@      .@      2@      &@      6@      2@      $@      2@      *@        
$
Conv1/biases/summaries/mean_1 ��=
&
Conv1/biases/summaries/stddev_1�u�;
#
Conv1/biases/summaries/max_1.��=
#
Conv1/biases/summaries/min_1��=
�
 Conv1/biases/summaries/histogram*�	   ���?   ��?      @@!   ��@)��m}��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      9@      @        
�#
Conv1/conv1_wx_b*�"	    #��   �-�?     $3A! k���@)G��c*��@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ����ž�XQ�þ�5�L�>;9��R�>5�"�g��>G&�$�>��~���>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�               @     �a@     �z@     ��@     l�@     �@     h�@     ��@     ~�@     ��@     ��@     ��@     �@     T�@     �@     u�@     ɳ@     ĳ@     
�@     �@     '�@     m�@     !�@     ߴ@     �@     �@     ��@     ��@     �@     Я@     <�@     ȫ@     ��@     ȩ@     �@     ��@     �@     �@     ��@     h�@      �@     ��@     p�@     ��@     $�@     Б@     ��@     @�@     ��@     ��@     ��@     H�@     p�@     0�@     �~@     �@      {@     �x@     x@     �r@     Pt@     �r@     �n@     �k@     �j@     @k@     `f@     �b@      d@     �`@      ^@     @Z@     @]@     �Y@      V@     �R@     �R@      R@     �M@     �K@     �H@      ?@      D@      ?@      6@      ;@      :@      4@      *@      "@      2@      *@      2@      3@      *@      @      "@      ,@      "@      *@      @      "@       @      @      @      @       @      @      @       @      @      @      �?      �?              �?       @               @       @               @              �?      �?      @              �?      �?              �?              �?              �?              �?               @      �?              �?               @              �?      �?              �?               @      @              �?       @       @              �?       @               @              @      �?       @      �?       @      @      @      �?       @      @      @      @      &@      "@      "@      *@      (@       @      2@      .@      2@      6@      6@      =@      9@      :@      ?@      <@      <@      E@     �J@      K@     �O@     �O@      K@     �N@      S@      W@     �T@      W@     �^@      `@     �`@     �a@     �e@     �i@     �e@     @j@      l@     @o@     t@     Pt@     �w@     pz@     p|@     �~@     `�@     ��@     ��@     ��@     ��@      �@     ��@     L�@     ��@     ��@     P�@     ̙@     ̜@     ��@     4�@     `�@     ��@     |�@     ̬@     g�@     ��@     ��@     �@     �@     ��@     ��@     ��@    `��@    ��!A    ���@    �l�@     ��@     ��@    ���@     ��@    ���@    ��@     ��@    ���@    ��@     ��@    ���@     G�@    ���@     '�@     0�@     ��@     ��@     Щ@     f�@     0�@     x�@     �|@     `c@      @@      .@        
�
Conv1/h_conv1*�   �-�?     $3A!��))TA)K��U�!�@2�	        �-���q=�5�L�>;9��R�>5�"�g��>G&�$�>��~���>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�	            �aA              �?              �?               @      �?              �?               @              �?      �?              �?               @      @              �?       @       @              �?       @               @              @      �?       @      �?       @      @      @      �?       @      @      @      @      &@      "@      "@      *@      (@       @      2@      .@      2@      6@      6@      =@      9@      :@      ?@      <@      <@      E@     �J@      K@     �O@     �O@      K@     �N@      S@      W@     �T@      W@     �^@      `@     �`@     �a@     �e@     �i@     �e@     @j@      l@     @o@     t@     Pt@     �w@     pz@     p|@     �~@     `�@     ��@     ��@     ��@     ��@      �@     ��@     L�@     ��@     ��@     P�@     ̙@     ̜@     ��@     4�@     `�@     ��@     |�@     ̬@     g�@     ��@     ��@     �@     �@     ��@     ��@     ��@    `��@    ��!A    ���@    �l�@     ��@     ��@    ���@     ��@    ���@    ��@     ��@    ���@    ��@     ��@    ���@     G�@    ���@     '�@     0�@     ��@     ��@     Щ@     f�@     0�@     x�@     �|@     `c@      @@      .@        
%
Conv2/weights/summaries/mean_1���
'
 Conv2/weights/summaries/stddev_1B�=
$
Conv2/weights/summaries/max_1'eT>
$
Conv2/weights/summaries/min_1��W�
�
!Conv2/weights/summaries/histogram*�	   �0�ʿ   ऌ�?      �@! ��P�)��ڿ̿x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ뾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾;�"�q�>['�?��>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �Q@     ؂@     ��@     ��@     D�@     ��@     �@     ��@      �@     $�@     �@     �@     H�@     `�@     ��@     0�@     ��@     ��@     ��@     ��@     P�@     p�@     �~@     `|@     x@     �y@      v@     Ps@     pr@     pq@      p@     �l@     �f@     �f@      e@     �b@     �a@     @]@     �W@     @R@     �[@      X@      R@     @T@     �P@     �O@      N@     �K@     �C@      C@      E@     �@@      >@      ;@      9@      .@      0@      5@      ,@      6@      0@       @      (@      @      *@      &@      &@      @      @      @      @      @      @      @       @       @      �?      @       @               @      �?       @      �?       @      �?      �?               @              �?               @       @              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?      �?              @      @      �?              �?              @               @              @       @      �?      @       @      @      @      @      @       @      @      @      @      @      @      @       @      $@      $@      @      ,@      *@      (@      $@      ,@      4@      3@      ;@      2@     �A@      @@      @@      D@     �I@     �E@      L@     �F@      K@      L@     �S@      S@      Y@     @T@     @V@      ]@     �_@     ``@      d@     �h@     `f@     `h@     �o@     �j@     �p@     Ps@     �s@     Pu@     �x@      y@     �~@     p~@     �~@     ��@     h�@     X�@     ��@     ȋ@     ��@     �@     ��@     �@     l�@     �@     ��@     0�@     �@     ��@     ��@     �@     �@     X�@     `@      A@        
$
Conv2/biases/summaries/mean_1B��=
&
Conv2/biases/summaries/stddev_1�+P;
#
Conv2/biases/summaries/max_1�F�=
#
Conv2/biases/summaries/min_1�־=
�
 Conv2/biases/summaries/histogram*q	   ��ڷ?   @ڨ�?      P@!   6�@)���rq��?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:              �M@      @        
�"
Conv2/conv2_wx_b*�"	   �s
 �   ����?     $#A! ��Ï��)�O�֤�@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(���E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�
�%W����ӤP���f^��`{>�����~>�����>
�/eq
�>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�               @      (@     �J@     �b@     s@     P�@     �@     8�@     N�@     ,�@     Į@     ��@     ��@     ��@     7�@     ?�@     ��@     %�@     �@    ���@     ��@     9�@    ���@    ��@    ���@    �d�@    �8�@    ��@    ���@     S�@     ��@     *�@     X�@     ��@     �@     չ@     �@     |�@     T�@     ��@     ܱ@     ��@     
�@     ��@     "�@     j�@     ^�@     J�@     h�@     ��@     ��@     ��@     �@     p�@     $�@     |�@     �@      �@     �@     �@     ��@     ��@     H�@     `�@     X�@     �s@     �s@     @s@      n@     ps@      m@     �d@     `f@     @b@     �f@      `@     �d@      Z@     @U@     @W@     �Q@      S@     �R@     @W@     �X@      A@      H@      B@      Z@      6@      8@      9@      9@      6@      5@      2@     �L@      0@      *@      $@      @@      &@      @      &@      @      @      "@       @      @       @       @      �?       @      �?      @              @      �?      @      �?      @              �?              @       @              �?      �?              �?               @               @              �?               @              �?              �?               @       @      �?       @      �?              �?       @       @              �?      @              @      �?      @      �?      @      @      @      @      @      @     �W@      @      @      @      &@      $@      2@     �X@      &@      2@      $@     �D@      K@      0@      ;@      :@      H@      ;@     �B@     @Q@     @Z@     �H@     �O@     @P@     @P@      P@     �V@      W@     �_@     @X@      `@     p@      e@     �a@     Pq@     �k@     �q@     �o@     �{@     �p@      s@     pw@     pz@     �{@     �y@     ��@     ��@     ��@     ��@     �@     h�@     �@     ̖@     t�@     D�@     L�@     l�@     p�@     �@      �@     l�@     �@     ��@     ��@     ֪@     ��@     ܯ@     ��@     ��@     ��@     &�@     �@     ��@     ��@    ���@     �@    ���@    ��@    ���@    �G�@     ��@     �@     N�@    �o�@    ���@     ��@    ��@     ��@    ���@     ��@    �(�@      �@     Q�@     �@     a�@     �@     ��@     $�@     l�@     ��@     {@     �k@     �W@      4@      @       @        
�
Conv2/h_conv2*�   ����?     $#A! ts�h�@)�q� ���@2�        �-���q=f^��`{>�����~>�����>
�/eq
�>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�            (�A              �?               @              �?              �?               @       @      �?       @      �?              �?       @       @              �?      @              @      �?      @      �?      @      @      @      @      @      @     �W@      @      @      @      &@      $@      2@     �X@      &@      2@      $@     �D@      K@      0@      ;@      :@      H@      ;@     �B@     @Q@     @Z@     �H@     �O@     @P@     @P@      P@     �V@      W@     �_@     @X@      `@     p@      e@     �a@     Pq@     �k@     �q@     �o@     �{@     �p@      s@     pw@     pz@     �{@     �y@     ��@     ��@     ��@     ��@     �@     h�@     �@     ̖@     t�@     D�@     L�@     l�@     p�@     �@      �@     l�@     �@     ��@     ��@     ֪@     ��@     ܯ@     ��@     ��@     ��@     &�@     �@     ��@     ��@    ���@     �@    ���@    ��@    ���@    �G�@     ��@     �@     N�@    �o�@    ���@     ��@    ��@     ��@    ���@     ��@    �(�@      �@     Q�@     �@     a�@     �@     ��@     $�@     l�@     ��@     {@     �k@     �W@      4@      @       @        

cross_entropy_scl��G>

training_accuracy�k?
����T      u���	���v���A�
*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�͐�Ja��(D����ha�x���wP��*X�`eia��@Jm�DH�"ib�����;��*���R<�p>�pf��@C�e��yQ�ʵ }p��z�4&��MX�n��N�灅�A3��;Ol���~4�❇�21��2I���8/]�����[��+O��������3�,C-�m��p.V��6<�gӭp\Z��9Iz�ܯ'8��<�r.��(���\� �#�|���n�������z�7}aju\���    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H  /IDAT(��ѭK�Q�ge21^�ů6�� 
�ô"�Z�ˌ���b1�"�Ɗ��2���Q���<��m��}g<����p����z���r�[9 )���Vy:?�P%�����8л����&]���̻��)�;y�<�m���ѫĳ��Er�Ff�*�J:^�G�xO�d)���w� ��w��x@ �1@���4���]��< ����ɂ�1 �s��"��2���f��\I��$#IF��qky��F��( ��[?���X}�j$���T��e_����VH����?m!��R�<�    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�͑?KBQ��V\C%'Epw�����.~ ��nJ����A����krhq+i	A�l����⟣�͡�r��s����M�yь����������@��id��0 �;Hx���T3M�
z�Y��)�$ HΤ����]n.�:��7�軶c���������9� _mϻ~Z�B�z݋t��0���8ma����"u�p"�m����z�� �}t���Ó���۷u��e	nGN�jŚ    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`��%�w���7��O�"���ϟ�����JS�Ɵ?��x���*^LI�nՈo��`���_�����償�����{���<b�C��۟M�8�*��e����*����s��|i�&Y�
��p։c���^w����m\V3��{�d������*\��~��G�����~��*���ϟ���k���ϟCq8,Ը~�KT� a`����    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�ő�uD!C5�Å�GB!L�>|R�3}�,�s0/��+��%�_�'��~�lB�0�dФ�N0��,�oؔdJR��kK�U&(ccy:�Q$糷�i��G�yM �7�b�Z��
f�+_Sj�	_���Io��ێ���y�O^�$^��|�py\|6����]�q��*��=�Q?��Z�t63�(    IEND�B`�
%
Conv1/weights/summaries/mean_1�LK�
'
 Conv1/weights/summaries/stddev_1���=
$
Conv1/weights/summaries/max_1�<K>
$
Conv1/weights/summaries/min_1UV�
�
!Conv1/weights/summaries/histogram*�	   @��ʿ   ��g�?      �@!  ���)-�EW�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�IcD���L��qU���I��!�A?�T���C?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @      @      *@      1@      .@      6@      9@      1@      2@      4@      5@      2@      *@      1@      *@      .@      1@      *@      1@      2@      &@      (@      @      @      @      @      "@      @      �?      @       @      @               @       @       @      @      �?      �?      @      �?              �?       @       @      �?              �?      �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?      �?       @      �?       @      @              �?              @       @      @      @      @       @      @      @      @       @      �?      "@       @      @      &@      $@      $@      @      (@      &@      .@      2@      (@      3@      3@      6@      $@      0@      2@      &@      6@      1@      (@      1@      *@        
$
Conv1/biases/summaries/mean_1���=
&
Conv1/biases/summaries/stddev_1}�;
#
Conv1/biases/summaries/max_1�2�=
#
Conv1/biases/summaries/min_1}ٺ=
�
 Conv1/biases/summaries/histogram*�	   �/[�?   �_&�?      @@!   ���@)%���-��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      9@      @        
�#
Conv1/conv1_wx_b*�"	    �7��    �|�?     $3A! *�J�_�@)�TR�;P�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%���>M|Kվ��~]�[Ӿ�XQ�þ��~��¾��|�~���MZ��K�����m!#���
�%W��39W$:��>R%�����>���]���>�5�L�>;9��R�>���?�ګ>5�"�g��>G&�$�>�[�=�k�>��~���>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?�������:�              @     Pr@     �@     0�@     ��@     H�@     Z�@     �@     ^�@     �@     ��@     �@     ϲ@     7�@     �@     d�@     ��@      �@     �@     ��@     y�@     ĵ@     \�@     �@     V�@     �@     T�@     ��@     l�@     :�@     (�@     J�@     ��@     X�@     ֨@     ̦@     Z�@     ��@     ڡ@     ��@     ؜@     ��@     p�@     ��@     @�@     Г@      �@     ��@     8�@     ��@     ��@     p�@     ��@     P�@      �@      @     p~@      {@      x@     �u@     �s@     0r@     �o@     �o@      n@      i@     �e@     �c@     @e@     �^@     �a@     �W@     �Y@     �[@     �S@     @S@     @R@     @R@      J@     �J@     �C@     �E@     �E@      ;@      A@      9@      ;@      8@      (@      4@      2@      ;@      2@      &@      3@      "@      ,@      0@       @      0@      $@      "@      @      @      @      @      @      @       @      @              @       @       @       @      @      �?       @      �?      �?      @      �?      �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @       @       @       @      �?       @      �?      �?      �?      �?       @               @      @      @      @       @      �?      @      @      $@      @       @      @       @      "@      @      $@      &@      @      ,@       @      *@      .@      *@      6@      =@      7@      8@      :@      C@      @@      ?@      D@      E@      J@      O@      L@     �Q@     @Q@      S@      Z@     @Y@     @_@     @Z@     �^@      a@     @e@     �e@     �g@     @h@      j@      n@     �q@     �r@      v@     `v@     �y@     �{@     @~@      @     x�@     0�@     0�@     ��@     Ȍ@     �@     @�@     <�@     D�@     �@     ��@     4�@     ܠ@     ڡ@     ޤ@     t�@     ,�@     ��@     &�@     �@     ��@     h�@     ��@    �q�@     }�@     ]�@    ��@    v� A    ���@     q�@     ��@    ��@    ���@     �@     +�@    �2�@     ��@    ��@     ��@     g�@    �F�@     �@    �E�@     ��@     �@     c�@     P�@     �@     B�@     d�@     p�@     ��@     �e@     �O@      0@      �?        
�
Conv1/h_conv1*�    �|�?     $3A! �ܾ	�A)�e��e��@2�        �-���q=39W$:��>R%�����>���]���>�5�L�>;9��R�>���?�ګ>5�"�g��>G&�$�>�[�=�k�>��~���>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?�������:�            �A              �?              �?              �?              �?              �?              �?      �?       @       @       @       @      �?       @      �?      �?      �?      �?       @               @      @      @      @       @      �?      @      @      $@      @       @      @       @      "@      @      $@      &@      @      ,@       @      *@      .@      *@      6@      =@      7@      8@      :@      C@      @@      ?@      D@      E@      J@      O@      L@     �Q@     @Q@      S@      Z@     @Y@     @_@     @Z@     �^@      a@     @e@     �e@     �g@     @h@      j@      n@     �q@     �r@      v@     `v@     �y@     �{@     @~@      @     x�@     0�@     0�@     ��@     Ȍ@     �@     @�@     <�@     D�@     �@     ��@     4�@     ܠ@     ڡ@     ޤ@     t�@     ,�@     ��@     &�@     �@     ��@     h�@     ��@    �q�@     }�@     ]�@    ��@    v� A    ���@     q�@     ��@    ��@    ���@     �@     +�@    �2�@     ��@    ��@     ��@     g�@    �F�@     �@    �E�@     ��@     �@     c�@     P�@     �@     B�@     d�@     p�@     ��@     �e@     �O@      0@      �?        
%
Conv2/weights/summaries/mean_1񖩺
'
 Conv2/weights/summaries/stddev_1T�=
$
Conv2/weights/summaries/max_1��S>
$
Conv2/weights/summaries/min_17�Y�
�
!Conv2/weights/summaries/histogram*�	   ��>˿   @�}�?      �@!��[O��P�)���x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��E��a�Wܾ�iD*L�پ��������?�ګ������~�f^��`{�E��a�W�>�ѩ�-�>I��P=�>��Zr[v�>�FF�G ?��[�?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �R@     ��@     ��@     ��@     \�@     ��@     �@     ��@     �@     $�@     $�@     ��@     �@      �@     ؍@     (�@     ��@     ��@     ��@      �@     �@     Ѐ@     �@     �{@     @x@      y@     �u@     �s@     �s@     �o@     p@     `m@      g@     �e@      e@      c@     �a@     �X@     @[@     @Y@     �X@     �S@     @S@     @P@     �P@     �S@      L@     �H@      G@      D@      ?@      @@      :@     �A@      <@      6@      6@      1@      0@       @      5@       @      $@      0@      @      (@      *@      @      "@      "@      (@      @      @      @      �?       @      �?      @       @      @      @      @      @      @       @              �?      �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?              @              �?              �?              @      @      @       @       @      �?       @      @      @               @      @       @      $@      "@      @      @       @      *@      $@      @      5@      .@      .@      *@      0@      7@      7@      8@      9@      <@      :@      F@      =@     �H@      E@     �F@     �G@     �P@     �P@     �S@     �S@      W@     @U@      T@     @[@     �a@     �^@      d@     �f@     �g@     �i@     �m@     �k@     �q@     �q@     �s@     @t@     �z@      z@      |@     `@      @     x�@     ��@     �@     x�@     ��@     ��@     ��@     H�@     �@     �@     P�@     ��@     0�@     �@     ��@     ��@     Ў@     @�@     �@     �@     �A@        
$
Conv2/biases/summaries/mean_17q�=
&
Conv2/biases/summaries/stddev_1'�Q;
#
Conv2/biases/summaries/max_1P<�=
#
Conv2/biases/summaries/min_1+M�=
�
 Conv2/biases/summaries/histogram*�	   `�ɷ?    �ǻ?      P@!   �&@)�ǒ?���?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?     �M@      @        
�#
Conv2/conv2_wx_b*�#	   శ�   �%# @     $#A! P�fg
��)|��t��@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ��uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ
�/eq
Ⱦ����ž�[�=�k���*��ڽ�5�"�g���0�6�/n���i����v>E'�/��x>��ӤP��>�
�%W�>���m!#�>��|�~�>���]���>�[�=�k�>��~���>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              �?      @      B@      Z@      n@     0z@     ��@     ��@     H�@     Τ@     F�@     �@     ��@     ��@     Ļ@     վ@     �@    �{�@     )�@     ��@     ��@     A�@    ��@     ��@     o�@    �j�@    ���@     ��@     ��@     �@    ���@     �@     �@     ��@     ��@     �@      �@     ˵@     t�@     ĳ@     l�@     ��@     ��@     ث@     d�@     ~�@     ��@     x�@     ��@     �@     <�@     ��@     ��@     8�@     ؔ@     D�@     �@     �@     �@     ��@     Љ@     ��@     �|@     �@     �y@      x@     �t@     �}@     �s@     �s@      o@      j@     @m@     �b@     `c@     �`@     @b@     �d@      [@     @R@     @]@     �V@     @U@     �N@     �Q@     �K@     �J@     �T@      C@      A@      ?@      ;@     �D@      =@     @Y@      :@      <@      ,@      (@      "@      (@      &@      ,@       @      $@      @      @      @      @      @      @      :@      @       @              �?      @      �?      �?       @      �?       @      �?       @              �?      �?              �?              �?      �?              1@              �?              �?              �?              �?      �?              �?              �?               @       @              �?      �?              @      �?      �?       @       @       @      @       @      �?      @       @              �?      @       @      �?      @      "@     �E@      @      @      $@      &@       @      0@      2@      1@      (@      =@      F@      .@      :@      >@      ;@      >@      <@      @@      E@     �F@     �J@      I@     @T@     @Z@     �n@     �[@     @]@      [@     �n@     �d@     �]@     �b@     �g@     �j@     0w@     �k@      v@      y@     �p@     �|@     Pt@     ��@     �~@     0�@     �|@     p�@     �@     �@     x�@     P�@     \�@     ,�@     ��@     4�@     P�@     ȗ@     T�@     ȟ@     ��@     ԣ@     �@     ��@     Ω@     ��@     *�@     ��@     �@     D�@     �@     �@     4�@     ��@     z�@    ���@     ��@    ���@     J�@     ��@     &�@     ^�@     ��@    �7�@    �5�@    �h�@     ��@     ��@    �!�@    ���@     ��@    � �@     �@     ��@      �@     �@     l�@     ��@     ��@     0�@     ��@     �~@     `p@     �_@      <@      "@      @      �?        
�
Conv2/h_conv2*�   �%# @     $#A! ���*�@)������@2�	        �-���q=�i����v>E'�/��x>��ӤP��>�
�%W�>���m!#�>��|�~�>���]���>�[�=�k�>��~���>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�	            ��A              �?              �?      �?              �?              �?               @       @              �?      �?              @      �?      �?       @       @       @      @       @      �?      @       @              �?      @       @      �?      @      "@     �E@      @      @      $@      &@       @      0@      2@      1@      (@      =@      F@      .@      :@      >@      ;@      >@      <@      @@      E@     �F@     �J@      I@     @T@     @Z@     �n@     �[@     @]@      [@     �n@     �d@     �]@     �b@     �g@     �j@     0w@     �k@      v@      y@     �p@     �|@     Pt@     ��@     �~@     0�@     �|@     p�@     �@     �@     x�@     P�@     \�@     ,�@     ��@     4�@     P�@     ȗ@     T�@     ȟ@     ��@     ԣ@     �@     ��@     Ω@     ��@     *�@     ��@     �@     D�@     �@     �@     4�@     ��@     z�@    ���@     ��@    ���@     J�@     ��@     &�@     ^�@     ��@    �7�@    �5�@    �h�@     ��@     ��@    �!�@    ���@     ��@    � �@     �@     ��@      �@     �@     l�@     ��@     ��@     0�@     ��@     �~@     `p@     �_@      <@      "@      @      �?        

cross_entropy_scl:��>

training_accuracy�(\?E�Y7T      �b�i	t�3y���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   IDAT(�c``���E 2�	�����Z��J�����Q%Qu���GX�g�n����?�Z�~�2ĩs#�ZT�oO�4qJ�9����\$#N����$Cڟ?gp�p�ڿ8%E��t T�%#�Y�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�ݑM� ����e�֛��,.��y�����|C�^����	�n&�m<��Rꡫ*�YPR��{	l��m�)��{R"zOPμ��gQIf ����!���!0��Oy��
���޶0�=��Ke�=�U�@j����    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c`P���������������f�r�on��x��݄M���
֞��j��```P}�P��L��5�%'��X$�������2o�����!	������-��﹅�1g��")���������~\�e`��B�d|� �K����oTq��=��wv)6�MՀqa!����/� ��CO&$~�H�b��ڿ�N�V��N� URm��N    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H  $IDAT(���;H�a��&iW��.SC�"F�4�t�-ھȡ-h�g��@�ƈg)pi���P�TR�lPL���g:�?/�4cffk�(�y�}N�|���y8R�}���|4��=��`�K�E�����Th�m �y��u��a���H�ܯu� &Ʊ=I�7�H�W��8�
��t%iq���e3�I)l���03[����%�-�[�^g�hp�.I��c� q��Fy�> �����g��ǤOLa@�t���$+I[E���)IN��t�*Y>��U}ѳ
&Cr�w�[�	�q?�z��b��O    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H  IDAT(�͐�+a�?��3���g�qQ.�88I9��(\����`O{wU�Ɂ��è&���gFf��U�^>��}ޞ��ҟ�\���Cc��w��{���L 8�a���p6�N��<F�k �-Y��'�P/�a�wn]K�g�:�-�������
��0��g.#�*��� �|�%mU�.EI��|Ƈ�@�zn�����J����������R]�M'^d�
Y���o(I��P㜤긯��8�٨I�++X��ᳲ�,�� �<��&������?���˨�NP��    IEND�B`�
%
Conv1/weights/summaries/mean_1רB�
'
 Conv1/weights/summaries/stddev_1Rӷ=
$
Conv1/weights/summaries/max_1 �K>
$
Conv1/weights/summaries/min_1 �V�
�
!Conv1/weights/summaries/histogram*�	    ��ʿ    �t�?      �@!  �|�)$X�]0�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��u�w74���82�nK���LQ?�lDZrS?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @       @      (@      0@      1@      6@      7@      2@      1@      5@      4@      3@      (@      2@      (@      0@      .@      0@      3@      *@      (@      (@      @      "@      @      @      "@      @       @      @      "@      �?       @      �?              @       @      �?      �?      @      �?      �?              �?      �?      @      �?               @      �?              �?              �?              �?              �?              �?               @              @      �?       @               @       @      �?      �?      �?      @      �?      @      @      @      @      @      @      @      @      @      "@      @      *@      (@      @      @      (@      "@      .@      2@      *@      2@      5@      6@      "@      0@      1@      (@      5@      2@      &@      2@      *@        
$
Conv1/biases/summaries/mean_1Қ�=
&
Conv1/biases/summaries/stddev_1.��;
#
Conv1/biases/summaries/max_1
��=
#
Conv1/biases/summaries/min_17�=
�
 Conv1/biases/summaries/histogram*�	   �C�?   @9�?      @@!   2Z�@)�uu}S~�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      8@      @        
�!
Conv1/conv1_wx_b*�!	   ��@��    L��?     $3A! ǘ9��@)��W�
�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE������>M|Kվ��~]�[Ӿjqs&\�Ѿ�XQ��>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              @      [@     pw@     ��@     ��@     ��@     ̜@     l�@     `�@      �@     �@     ��@     ��@     ��@     :�@     K�@     ��@     �@     ��@     �@     ��@     E�@     ��@     V�@     s�@     ݴ@     �@     G�@     �@     ΰ@     �@     �@     �@     ��@     *�@     ��@     �@     �@     ��@     ��@     p�@     ��@     0�@     p�@     ��@     x�@     t�@     đ@      �@      �@     Љ@     �@     �@     ��@     ��@     @�@      ~@     p{@     Pv@      w@     �s@     �r@     �l@     �i@     �n@      i@     @g@     �f@     �b@     @c@     �`@     �^@     �Z@      W@      X@     @T@      R@     �R@      O@     �I@     �J@     �E@      A@      A@     �D@      >@      6@      A@      ;@      0@      8@      0@      0@      *@      .@      &@      "@       @      *@      "@      @       @       @      @      @      �?      @      @      �?      @      @              �?      @      @               @      @       @       @              �?       @       @      �?              �?      �?              �?      �?              �?      �?      �?      �?              �?      �?      �?      �?      �?      �?              �?               @      �?       @               @      @       @       @      @      @      "@      @      @      @       @      $@       @      @      "@      @      *@      (@      "@      .@      ,@      &@      4@      5@      3@      8@      ;@      <@      >@      A@      F@     �F@     �K@     �I@      K@      P@     @U@     �W@     �R@     �U@     �[@     �\@     �a@     @a@     �b@      f@     �f@     �j@     `l@      o@     �p@     0s@     �t@     pv@     �y@     �z@     �@     �@     0�@     Є@     �@     ��@     ��@     �@     L�@     Ȓ@     ��@     $�@     ��@     ܝ@     �@     b�@     F�@     n�@     r�@     ®@     �@     G�@     4�@     �@     P�@    ���@    �H�@    ���@    ��@    8 A    ���@     9�@     `�@    �P�@    ��@    �e�@    ��@     �@     ��@    �4�@    ���@     �@    ���@     ��@     ��@     r�@     ��@     ��@     ı@     2�@     �@     0�@     ��@     (�@     �b@     �A@      @        
�
Conv1/h_conv1*�    L��?     $3A!�����A)�i=�1��@2�        �-���q=�XQ��>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            @�A              �?      �?              �?      �?      �?      �?              �?      �?      �?      �?      �?      �?              �?               @      �?       @               @      @       @       @      @      @      "@      @      @      @       @      $@       @      @      "@      @      *@      (@      "@      .@      ,@      &@      4@      5@      3@      8@      ;@      <@      >@      A@      F@     �F@     �K@     �I@      K@      P@     @U@     �W@     �R@     �U@     �[@     �\@     �a@     @a@     �b@      f@     �f@     �j@     `l@      o@     �p@     0s@     �t@     pv@     �y@     �z@     �@     �@     0�@     Є@     �@     ��@     ��@     �@     L�@     Ȓ@     ��@     $�@     ��@     ܝ@     �@     b�@     F�@     n�@     r�@     ®@     �@     G�@     4�@     �@     P�@    ���@    �H�@    ���@    ��@    8 A    ���@     9�@     `�@    �P�@    ��@    �e�@    ��@     �@     ��@    �4�@    ���@     �@    ���@     ��@     ��@     r�@     ��@     ��@     ı@     2�@     �@     0�@     ��@     (�@     �b@     �A@      @        
%
Conv2/weights/summaries/mean_1����
'
 Conv2/weights/summaries/stddev_1 �=
$
Conv2/weights/summaries/max_1��S>
$
Conv2/weights/summaries/min_1��Z�
�
!Conv2/weights/summaries/histogram*�	   `PT˿   @�x�?      �@!��氪�P�)?ӚR�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �R@      �@     ��@     �@     d�@     ��@     �@     ��@     ܓ@     �@     <�@     ̒@     `�@     ��@     ��@     p�@     `�@     ��@     h�@     ��@     Ђ@     ��@     �@     `|@      y@     �x@     �u@     �s@     Ps@     `p@     �p@     �k@      g@     �e@     �e@     �c@     �^@     @^@     �X@      [@      V@     �S@     @T@     �Q@     �S@      P@      O@     �D@     �C@      D@      F@      >@      ?@      >@      >@      @@      ,@      &@      3@      .@      0@      $@      $@      (@      @      &@      @      @      @      $@       @      @      @      @      @      @       @      �?              �?      @      �?       @      @      @      �?              �?      �?              �?      �?              �?              �?      �?      �?              �?       @      �?              �?      �?              �?      �?       @      �?       @              �?      @      @              �?              �?      �?      @      @       @      @      @      @       @      @      "@       @      @      @      @      *@      @      "@      .@      &@      &@      "@       @      0@      8@      5@      8@      D@      ;@      ?@      A@      C@      F@      E@      K@     �I@     �M@      Q@     @P@     �N@     �W@     �V@      V@     @]@     �`@      a@     `c@     `g@     @e@     �k@      n@      m@     `p@      q@      u@     0u@      y@     �z@     �|@     p@     `@     X�@     �@     �@     (�@     �@     ȋ@     Ќ@     ��@      �@     X�@     4�@     Ԓ@     �@     $�@     ��@     Б@     ؎@     �@     8�@     P@      B@        
$
Conv2/biases/summaries/mean_1bL�=
&
Conv2/biases/summaries/stddev_1�S;
#
Conv2/biases/summaries/max_1/��=
#
Conv2/biases/summaries/min_1Q�=
�
 Conv2/biases/summaries/histogram*�	    j��?   �E�?      P@!   .�	@)4h$i��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @     �L@      @        
�%
Conv2/conv2_wx_b*�%	    �� �   @���?     $#A! (7����)�����@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(�E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þG&�$��5�"�g�����n�����豪}0ڰ���������?�ګ�39W$:���.��fc���ہkVl�p>BvŐ�r>�5�L�>;9��R�>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              @      1@     @U@     �j@     �x@     H�@     ؒ@     ��@     H�@     �@     E�@     ��@     =�@     ��@      �@    ���@     ��@     S�@     ��@    ���@     <�@     �@    �=�@     ��@     ��@     e�@    �w�@     ��@     ��@     ��@     �@     ��@     ��@     ��@     q�@     |�@     ��@     ��@     �@     ��@     |�@     ��@     n�@     ԫ@     &�@     ��@     ��@     v�@     ��@     p�@     H�@     ��@     4�@     @�@     (�@     �@     ��@     ��@     ��@     `�@     Ђ@     ؃@     ��@     �~@     �|@     ��@     Pr@     @r@     �r@     �n@     @l@      f@      h@      g@     `f@     �X@      h@      `@     @W@     �]@     �S@     �P@      P@     �U@     @^@     �H@      N@     �H@      <@      @@     �@@      <@      ?@      6@      1@      6@      >@      $@      *@      $@      2@      "@      $@       @      @      @      @      @      @      @      @     �C@      @      @      @      @     �C@       @      @       @              �?              �?      �?      �?              �?               @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @       @       @              �?              �?       @      �?      @       @      �?       @      �?      �?      @      @      @      @      �?      @      @      @      $@      @      @       @       @      &@      "@      &@      *@      8@      &@      &@      3@      1@      2@      8@     �B@      7@     �K@      ?@      ;@      ;@      I@     �M@      K@      L@     �J@      N@      P@      T@     �P@     �[@     @Y@      ]@     `b@     `i@     @f@     �c@     �k@     �l@     p@     �r@      w@     p}@     8�@     `x@      |@     �{@     ��@     ؃@     �@     h�@     `�@     �@     ��@     (�@     P�@     @�@     �@     L�@     ,�@     �@     R�@     J�@     ^�@     �@     �@     ĭ@     v�@     ��@     ��@     ˶@     *�@     ��@     ɺ@     ��@     ,�@    � �@     �@    ���@     ��@     ��@    ���@     ��@    ��@     ��@    ���@     F�@    ���@     ��@     ��@    �F�@     $�@    ���@     ��@     �@     ;�@     =�@     ��@     ʦ@     �@     <�@     `�@     p}@     `m@     �Y@      >@      @      @        
�
Conv2/h_conv2*�   @���?     $#A! �j#:��@)e%���@2�	        �-���q=ہkVl�p>BvŐ�r>�5�L�>;9��R�>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�	            �A              �?              �?              �?      �?              �?              �?              �?              �?               @       @       @              �?              �?       @      �?      @       @      �?       @      �?      �?      @      @      @      @      �?      @      @      @      $@      @      @       @       @      &@      "@      &@      *@      8@      &@      &@      3@      1@      2@      8@     �B@      7@     �K@      ?@      ;@      ;@      I@     �M@      K@      L@     �J@      N@      P@      T@     �P@     �[@     @Y@      ]@     `b@     `i@     @f@     �c@     �k@     �l@     p@     �r@      w@     p}@     8�@     `x@      |@     �{@     ��@     ؃@     �@     h�@     `�@     �@     ��@     (�@     P�@     @�@     �@     L�@     ,�@     �@     R�@     J�@     ^�@     �@     �@     ĭ@     v�@     ��@     ��@     ˶@     *�@     ��@     ɺ@     ��@     ,�@    � �@     �@    ���@     ��@     ��@    ���@     ��@    ��@     ��@    ���@     F�@    ���@     ��@     ��@    �F�@     $�@    ���@     ��@     �@     ;�@     =�@     ��@     ʦ@     �@     <�@     `�@     p}@     `m@     �Y@      >@      @      @        

cross_entropy_scl���=

training_accuracy��u?�hO�T      >�	a�e{���A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�c`���t�r~_���%����<\r�/��g�q�P��D�š���ߟ��L=���z\r�>b�%y���GH\ɉ20L����߿$p������Y �X�rV��ph����\6N����2��XE��wqhT������?o��ee�߿l����I�  ��8R`n%�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�&��p���Hǿ�v�$W���������V�a�����G'v9���~��0S�׿�8�t7��؈�L��~��'���EWV�LAl�	���j?���+L�Ȼ�����������3�Xkӏ0t�������������"����~F00000���M����G��.ӟ��*���~23|����l���p��3�>������|xAt�>dpPI�`��~��_h6:�I���� ��i&��&�    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c`6������s�$�_XYYYYu���AV!���?$ ga```�`ƥ���`��*q��|�{����gg3�K
1��)�������� 8�:�9�![p��p��6N����pJr��c'û�x$_��)���C�a��n��g�S'� �7��<(\��k,pۉ
 
0	�#S
    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�͑�m1�׆�4@���b�:�� �����T���<���^���������{��a�r}Һ렗n,]iΜ9+~���|bs�y�	�x!�%;C�i#�Q�m��r	���z`_��6z���Sb��`�X��P�G�:���EDB�9z����߮"���y�o��i%ډH<���@�A��ۤ<���,��*e����M��������_y`����    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(���1KBa�����:形\�k�~��mE-A��P���M�S�pk�֦��6�-:�Cb��C���}���9�缜����LN�9߰z*I��n�������y���߸����?Bغy�:���������FxH������;�2@)a|�k�~��W�E��+ jL��/E�G ��Ӄ�����Bw&�ײ�7ԇk��|I��S�$�b6��YO��ʶ��g=8Z�8    IEND�B`�
%
Conv1/weights/summaries/mean_1.�H�
'
 Conv1/weights/summaries/stddev_1_�=
$
Conv1/weights/summaries/max_1W�K>
$
Conv1/weights/summaries/min_1�W�
�
!Conv1/weights/summaries/histogram*�	   ���ʿ   �*x�?      �@!  ;�И�)�_���@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�IcD���L��qU���I��u�w74?��%�V6?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @       @      (@      0@      1@      5@      9@      2@      0@      5@      5@      4@      &@      0@      ,@      .@      .@      .@      2@      0@      &@      (@      @      @      @      @      @       @      @      @      @       @      �?      �?      @      �?      @              @       @               @      �?      �?              �?       @      �?       @      �?      �?              �?              �?              �?              �?      �?              �?      @      �?      �?       @              �?      �?       @      �?      @      �?      @      @      @      @      @      @      @      @      @      @      @      $@      "@      (@      @      @      *@      $@      ,@      2@      ,@      1@      4@      7@      "@      0@      2@      &@      7@      0@      (@      1@      *@        
$
Conv1/biases/summaries/mean_1La�=
&
Conv1/biases/summaries/stddev_1j�;
#
Conv1/biases/summaries/max_1t�=
#
Conv1/biases/summaries/min_1�E�=
�
 Conv1/biases/summaries/histogram*�	   ���?   ��!�?      @@!   �)�@)�
z��s�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      4@      @        
�"
Conv1/conv1_wx_b*�!	   ��/��    x��?     $3A! ��=���@)28�$[�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ;9��R���5�L��
�}���>X$�z�>�[�=�k�>��~���>
�/eq
�>;�"�q�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              *@     `b@      |@     ��@     ��@     ��@     4�@     �@     ʥ@     Χ@     ��@     f�@     ��@     ϱ@     ��@     ��@     /�@     ��@     ��@     ��@     ĵ@     �@     ,�@     ��@     ^�@     c�@     ��@     ?�@     �@     i�@     �@     ��@     ��@     ��@     �@     ܥ@     $�@     d�@     ��@     �@     ��@     @�@     ��@     ��@     4�@     L�@     ��@     (�@     ؋@     `�@     ��@     x�@     H�@     8�@     h�@     p|@     �z@     �y@     �v@     `t@     �q@      q@     �m@     `k@      g@     �g@     �c@      c@      c@      ^@     �_@     �[@     @[@      U@     �S@     @Q@     @Q@      L@     �N@     �I@      A@      G@      8@      3@      :@      <@      :@      *@      7@      =@      2@      <@      .@      .@      1@      (@      $@       @      "@       @       @      &@      @      @      @              @       @       @       @      �?      �?      @      @      @       @              �?      �?               @      �?              �?               @       @              �?              �?              �?              �?              �?              �?              �?       @              �?               @      @      �?      @      �?      @      �?      �?      @       @      @       @      @       @      @      @      @      @      @      @      @      @      "@       @      $@      $@      *@      2@      ,@      3@      9@      .@      <@      >@      =@      <@      5@      @@     �B@     �F@     �G@      H@      N@     �R@      U@      R@     �R@      V@     @V@     �\@      `@     �a@     �b@      b@     �e@     �f@      m@     `m@      n@     �p@      s@     `u@     Pw@      y@     �~@      �@     ؀@     x�@     ��@     (�@     �@     ،@     ��@     �@     D�@     �@     4�@     X�@     �@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     y�@     ��@     ��@    �-�@     ��@    XA    �yA    p��@     �@    ��@    ���@    �)�@    �X�@    � �@     ��@     �@     ��@    ���@     �@    ���@     ��@     v�@     �@     $�@     !�@     �@     f�@     ��@     |�@     �@     �@     @`@     �@@      @        
�
Conv1/h_conv1*�    x��?     $3A! �y�-pA)7N�M��@2�        �-���q=
�}���>X$�z�>�[�=�k�>��~���>
�/eq
�>;�"�q�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            �A              �?              �?              �?              �?       @              �?               @      @      �?      @      �?      @      �?      �?      @       @      @       @      @       @      @      @      @      @      @      @      @      @      "@       @      $@      $@      *@      2@      ,@      3@      9@      .@      <@      >@      =@      <@      5@      @@     �B@     �F@     �G@      H@      N@     �R@      U@      R@     �R@      V@     @V@     �\@      `@     �a@     �b@      b@     �e@     �f@      m@     `m@      n@     �p@      s@     `u@     Pw@      y@     �~@      �@     ؀@     x�@     ��@     (�@     �@     ،@     ��@     �@     D�@     �@     4�@     X�@     �@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     y�@     ��@     ��@    �-�@     ��@    XA    �yA    p��@     �@    ��@    ���@    �)�@    �X�@    � �@     ��@     �@     ��@    ���@     �@    ���@     ��@     v�@     �@     $�@     !�@     �@     f�@     ��@     |�@     �@     �@     @`@     �@@      @        
%
Conv2/weights/summaries/mean_1���
'
 Conv2/weights/summaries/stddev_1�$�=
$
Conv2/weights/summaries/max_1�dT>
$
Conv2/weights/summaries/min_1re\�
�
!Conv2/weights/summaries/histogram*�	   @��˿   ����?      �@!����P�)T^���x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���f�����uE���⾄iD*L�پ�_�T�l׾�[�=�k���*��ڽ�ہkVl�p>BvŐ�r>��~���>�XQ��>�����>
�/eq
�>jqs&\��>��~]�[�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �S@      �@     ��@      �@     d�@     ��@     �@     Г@     ��@     �@     (�@     ܒ@     d�@     @�@     ��@     p�@     ��@     8�@     ȅ@     x�@     (�@     ��@      @     �|@     `x@     �x@     �u@     �t@     �r@     �p@     Pp@     `j@      i@     �d@     �g@     �b@      _@     �[@     @X@      [@      W@      U@      U@     �P@     �P@     @P@     �Q@     �J@     �F@     �B@      ?@      A@      ;@      8@      ?@      0@      5@      7@      $@      &@      2@      *@      1@      2@      "@      .@      @      @      @      @       @       @      @      @       @       @       @      @      @       @       @      @      �?      @       @      �?      �?       @              �?      �?      �?      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @       @              �?      �?      @              �?      @      @              �?      @      �?      @       @      @      @      @              @      @      1@      @       @      &@      @      $@      (@      ,@      ,@      *@      6@      3@      4@      3@      :@      ;@     �B@      A@     �E@     �@@      J@     �@@     �K@      Q@      Q@      N@      T@     �S@     @T@     �U@     `a@     �_@     �`@      d@     �e@      g@     �k@      m@      l@     �p@      r@     Pt@     �t@     �y@     �z@     �|@     �~@     �@     P�@     ��@     ��@     (�@     ��@     ��@     ��@      �@     �@     ��@     \�@     �@     ,�@     ,�@     ��@     ��@     ��@     ��@     ��@     P@     �C@        
$
Conv2/biases/summaries/mean_1G+�=
&
Conv2/biases/summaries/stddev_1�Y;
#
Conv2/biases/summaries/max_1(��=
#
Conv2/biases/summaries/min_1���=
�
 Conv2/biases/summaries/histogram*�	   �Ԟ�?    �ػ?      P@!  ��h@)�^�=��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @     �L@      @        
�$
Conv2/conv2_wx_b*�$	    � �   �G� @     $#A! �!�k���)���ܝ��@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ
�/eq
Ⱦ����ž�XQ�þ��~��¾.��fc��>39W$:��>0�6�/n�>5�"�g��>�XQ��>�����>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              @      ?@     �Z@      f@     �y@     ��@     ��@     ��@     B�@     z�@     G�@     N�@     E�@     '�@     )�@    �A�@     ��@     ��@     ��@    �b�@     ��@     ��@    �a�@     �@    �D�@     6�@    ���@    �H�@    �3�@     L�@     ��@    ��@    ���@     �@     ��@     %�@     �@     ��@     �@     6�@     �@     ��@     ��@     H�@     ��@     ��@     �@     ��@     `�@     L�@     h�@     Ș@     `�@     �@     h�@     ��@     L�@     ��@     ��@     ��@     ��@     ��@     �~@     Ѕ@     �z@     �@     Py@     �x@     0q@     �l@      s@     @e@     �p@     `a@      j@     �c@     `h@      `@      S@     �Y@      T@      T@     `o@      P@      K@     �S@     �H@     �@@      <@     �A@      =@      :@      5@      6@      ;@      G@      D@      F@      *@      *@      (@      @      "@      @      @      $@      @      $@      @       @      @      @      @      @      �?       @      �?      �?      @              �?      �?      @       @      �?      �?              �?      �?      �?      �?              �?               @      �?              �?              �?              �?              �?               @              �?              �?               @               @              @      �?       @      @      �?       @      �?      @       @      @      �?      @      �?      c@       @      @       @      @      @      @       @      "@      @      "@      $@      $@      (@      0@      (@      ,@      &@      0@      2@      5@      <@     �@@      1@      A@      >@      D@     �C@     �D@     @T@     �H@      i@      M@     �U@     �b@     �S@      `@     �X@      Z@     �a@     �c@     �f@     `c@     �g@     @f@     @j@     �p@     �s@     �r@      u@     py@     w@     �@     8�@     (�@     x�@     ��@     H�@     Ȏ@     Ȏ@     ��@     �@     ,�@     Ė@     @�@      �@     (�@     �@     ��@     *�@     ��@     ��@     ʭ@     ��@     F�@     ��@     y�@     �@     Ҹ@     ��@     ��@     ɽ@     :�@    ���@     k�@     K�@     ��@     ��@     �@     ��@    ���@    �N�@    �"�@     9�@    ���@     ��@     �@     b�@     ��@     ~�@     �@     �@     ׵@     ��@     ث@     ��@     X�@      �@     0�@     �x@     @i@     �V@      >@      @              �?        
�
Conv2/h_conv2*�   �G� @     $#A! h��@)3�p�O3�@2�	        �-���q=.��fc��>39W$:��>0�6�/n�>5�"�g��>�XQ��>�����>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�	            �A              �?               @              �?              �?               @               @              @      �?       @      @      �?       @      �?      @       @      @      �?      @      �?      c@       @      @       @      @      @      @       @      "@      @      "@      $@      $@      (@      0@      (@      ,@      &@      0@      2@      5@      <@     �@@      1@      A@      >@      D@     �C@     �D@     @T@     �H@      i@      M@     �U@     �b@     �S@      `@     �X@      Z@     �a@     �c@     �f@     `c@     �g@     @f@     @j@     �p@     �s@     �r@      u@     py@     w@     �@     8�@     (�@     x�@     ��@     H�@     Ȏ@     Ȏ@     ��@     �@     ,�@     Ė@     @�@      �@     (�@     �@     ��@     *�@     ��@     ��@     ʭ@     ��@     F�@     ��@     y�@     �@     Ҹ@     ��@     ��@     ɽ@     :�@    ���@     k�@     K�@     ��@     ��@     �@     ��@    ���@    �N�@    �"�@     9�@    ���@     ��@     �@     b�@     ��@     ~�@     �@     �@     ׵@     ��@     ث@     ��@     X�@      �@     0�@     �x@     @i@     �V@      >@      @              �?        

cross_entropy_sclt`1>

training_accuracy�k?3��W[U      5+xq	��}���A�*˪
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H  IDAT(�c`�Y"g�_�!.%���!���
S�ª?�����e�*L�.yI�?õ���Ԩ����Nm�a����� �A��{�)�fp�gXt����?n�x����?��u@����p��?~zc�\���� ���o?��ógY0����ٱ��������W&��_ޢ	�jCJ���C���UeM��7M��_n��/fX��S�����Z����<>2D�%l�j��<��_����ϟ�f����)Z� ����i~a� ZrDB�B    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`�@z�_7�rW���%���9� ������Y�;X�h������!��L��g�a�Lg��g```�����z�P&W�	����=���ݜ����H��?[^����ϟ?!��������ޫ���]���}όM���A���~$.��_�8%v0��-iȰ��8�9�����=N��8u��{_�$�  � Eq㏍L    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c``������8�Do��^���d�U,��j�J��~#�����1L�	I�ӆ�?x��j�������6��lf�mT�J�z��� sv?V�ۿq�
<�㳠XZ����|>V�00ȼ���_(v9����}�j�U��пcݰk,����=r��'~E�+��3<�Ůs��'�ڰ�<�p�J��j�:��y��(������ ̏K��[o�    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H  IDAT(��б+�a���r:�d &�"e�E"2\)�,�%�m����n�I1�JI� �o3=a�e�9����a8\��z6�����z��������v܋9��,��{ ���qK�Z;�(/�y;��S�K���w���Z⌰���bq��X���5�0�s{���j�~v�f�����h�5�$]ڋf�:���Lԧ��A3��%#-C{5�i�$u�Zw�|)�r��D��[�x�p�@�P)�	�u�� ,�N�d�;��Ь����S���/��i�J	,�    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`��~�[�������.8��C*3.i���
p[�}�onY��,�LL�
�^;a2aj킳�H��,������,��Ý���PU��}���߿�g��,e�P��^^��Qך���Vɵ{(:���g````U:��߿_ڠH��g````����Y?o��w�\X��?i0000x|�'��������_3\���d֊_��`��d�Kr�y]]��KB(m��o�V�� �sRL�3��    IEND�B`�
%
Conv1/weights/summaries/mean_1�B�
'
 Conv1/weights/summaries/stddev_1��=
$
Conv1/weights/summaries/max_1��K>
$
Conv1/weights/summaries/min_10'X�
�
!Conv1/weights/summaries/histogram*�	    �˿    Rz�?      �@! ���x�)�P݊��@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�
����G�a�$��{E���%�V6��u�w74�1��a˲?6�]��?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @      "@      &@      .@      3@      5@      8@      2@      0@      6@      5@      0@      .@      1@      *@      *@      1@      (@      3@      0@      (@      &@      @      @      @      @       @       @       @      @      @      @      �?       @       @       @      �?              �?      @      @      �?              �?      �?              �?      �?      �?       @      �?              �?              �?              �?              �?      �?              �?       @       @              �?       @      �?               @      �?      @              �?      @      @      @       @      @      @      "@       @      �?       @      @      @       @      (@      $@      @      @      &@      (@      .@      1@      ,@      3@      3@      6@      $@      .@      3@      (@      6@      0@      (@      2@      (@        
$
Conv1/biases/summaries/mean_1vj�=
&
Conv1/biases/summaries/stddev_1��;
#
Conv1/biases/summaries/max_1��=
#
Conv1/biases/summaries/min_10�=
�
 Conv1/biases/summaries/histogram*�	   ��?   � �?      @@!   �N�@)�����u�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @      2@      @        
�"
Conv1/conv1_wx_b*�"	   @�r��   �m\�?     $3A! C��@)�$��^�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾�XQ�þ��~��¾;9��R�>���?�ګ>��n����>�u`P+d�>�[�=�k�>��~���>�XQ��>�����>;�"�q�>['�?��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              "@      ]@     pz@     �@     h�@     ؔ@     ��@     ��@     ��@     p�@     ��@     ª@     R�@     7�@     �@     ��@     Ͳ@     ߳@     ӳ@     �@     ��@     �@     �@     A�@     \�@     ۲@     X�@     �@     ��@     �@     .�@     �@     ��@     ��@     ��@     �@     ֣@     �@     ޠ@     L�@     �@     ��@     ��@     L�@     (�@     �@     <�@     �@     ��@      �@     Ȉ@     ��@     ��@     Ȃ@     �@     @�@     �|@     �v@     Pu@     Ps@      t@     �p@      o@     �n@     @k@     �j@      e@      c@     �`@     �^@      _@     �]@      V@      Y@      R@      T@     �R@      Q@     �K@     �H@      E@      D@      B@      >@      ;@      4@      <@      :@      <@      2@      ,@      3@      *@      *@      2@      1@      $@       @      @      "@      $@      @      @      @      @      @      @      @       @      @       @      �?      @       @       @       @              �?      �?               @      @      @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?               @       @      �?              �?      �?               @       @      �?      �?      �?       @      �?      @      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      ,@      0@      1@      (@      4@      3@      0@      <@      ;@      ;@      @@     �@@      9@      ?@      G@      J@     �D@     �J@     �M@     �N@      Q@     �U@      U@     @Z@     �X@      ]@      a@      a@     �d@      e@     �d@     �l@     @l@     �p@     �r@     �r@     v@     �x@      z@     @}@     @�@      �@     X�@     Ѕ@     ��@     ��@     ��@     ��@     8�@     l�@     d�@     p�@     ��@     ��@      �@     ��@     ��@     �@     ��@     ��@     �@      �@     l�@     �@    ���@    ���@     ��@    �3A    �AA    �A    �v�@     5�@    ��@     W�@    ���@    ���@    ���@     p�@    ���@    ���@    �Y�@     ��@     ��@     ��@     ��@     F�@     G�@     "�@     @�@     �@     �@     �@     �|@     �_@      >@       @        
�
Conv1/h_conv1*�   �m\�?     $3A!�KX�OA)��Tc���@2�        �-���q=;9��R�>���?�ګ>��n����>�u`P+d�>�[�=�k�>��~���>�XQ��>�����>;�"�q�>['�?��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            ��A              �?              �?              �?              �?              �?               @       @      �?              �?      �?               @       @      �?      �?      �?       @      �?      @      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      ,@      0@      1@      (@      4@      3@      0@      <@      ;@      ;@      @@     �@@      9@      ?@      G@      J@     �D@     �J@     �M@     �N@      Q@     �U@      U@     @Z@     �X@      ]@      a@      a@     �d@      e@     �d@     �l@     @l@     �p@     �r@     �r@     v@     �x@      z@     @}@     @�@      �@     X�@     Ѕ@     ��@     ��@     ��@     ��@     8�@     l�@     d�@     p�@     ��@     ��@      �@     ��@     ��@     �@     ��@     ��@     �@      �@     l�@     �@    ���@    ���@     ��@    �3A    �AA    �A    �v�@     5�@    ��@     W�@    ���@    ���@    ���@     p�@    ���@    ���@    �Y�@     ��@     ��@     ��@     ��@     F�@     G�@     "�@     @�@     �@     �@     �@     �|@     �_@      >@       @        
%
Conv2/weights/summaries/mean_1���
'
 Conv2/weights/summaries/stddev_1�(�=
$
Conv2/weights/summaries/max_10�U>
$
Conv2/weights/summaries/min_1�]�
�
!Conv2/weights/summaries/histogram*�	   ��˿    ���?      �@!@d7p�,R�)��w�	�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(���G&�$�>�*��ڽ>�[�=�k�>�����>
�/eq
�>��(���>a�Ϭ(�>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �S@     �@     ��@     �@     x�@     ��@     �@     �@     ؓ@     �@     <�@     ��@     ��@     ��@     ��@     ،@     ��@     ��@     0�@     H�@     Ȃ@     ؀@     �~@     �{@     �x@     �y@     0u@     �t@     @r@      p@     �p@     �l@     `f@      g@     �f@     `b@     �]@     @[@     �]@      Y@     �V@     �U@     @W@     @P@     �L@      P@     �P@      L@      C@      F@      @@      A@      >@      3@      3@      :@      3@      6@      3@      *@      0@      ,@      $@      *@      "@       @      $@      "@      (@      @      @      "@       @              @              @      @       @      @              @              �?      �?              �?      �?      �?       @               @               @      �?      �?      �?              �?              �?              �?      �?              �?              �?               @              �?              �?      @      �?              �?      @       @              �?      �?       @              �?       @      �?      �?       @       @      @       @      @      *@      @      @      @      @      @      @      (@       @      ,@      (@      8@      1@      7@      7@      ;@      @@      A@      =@      D@      7@     �D@      G@      G@      F@     @Q@      J@      O@     �Q@      [@     �R@     �X@     @^@     `b@     �`@      c@      c@      h@     �k@     �o@     @k@     �p@     @q@     �s@     �v@     Px@     pz@     P}@     P}@     P@     ��@     p�@     ��@     (�@     X�@     H�@     H�@     �@      �@      �@     8�@     �@      �@     \�@     `�@     �@     X�@     P�@     0�@     �~@      D@        
$
Conv2/biases/summaries/mean_1^��=
&
Conv2/biases/summaries/stddev_1�-X;
#
Conv2/biases/summaries/max_1@j�=
#
Conv2/biases/summaries/min_1���=
�
 Conv2/biases/summaries/histogram*�	   �_��?    H��?      P@!   ��@)���⦊�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @      M@      @        
�$
Conv2/conv2_wx_b*�$	   @DO�   @S�?     $#A! �.����)�~�(k��@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾K+�E��Ͼ['�?�;;�"�qʾ�*��ڽ�G&�$��.��fc��>39W$:��>���?�ګ>����>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              �?      @      4@     �]@     @n@     �}@     ��@     \�@     x�@     :�@     
�@     /�@     ��@     b�@     $�@     �@     h�@     ��@    ���@     ��@     ��@     I�@    ���@     0�@    �,�@     ��@     ]�@     ��@    ���@    �C�@    �N�@    ��@    �\�@    ��@     ��@     P�@     q�@     ��@     ��@     ��@     0�@     ��@     հ@     *�@     �@     ^�@     2�@     ��@     ��@     Ơ@     ě@     ��@     ��@     ę@     T�@     ��@     �@     ��@     `�@     `�@     ��@     ��@     P�@     @�@     p�@     �z@     �u@      w@     �u@     @u@     p}@      j@     �b@     �d@     �e@      t@      m@     �a@     �Y@     0s@     �S@     �Q@     �N@     @]@     @b@      F@     �H@     �L@      I@      >@      ?@     �J@      6@     �i@      9@      7@      ,@      3@      1@      3@      ,@     @k@      $@      $@      $@      "@      $@      "@      @      @      @      "@       @      @      @      @      @      @      �?       @      �?       @       @       @       @      �?       @               @              �?      �?      �?      >@              @              �?      �?              �?              �?              �?              �?              �?      �?              �?      �?      �?       @              �?      �?       @              �?              �?       @       @              @       @      @      B@      �?      �?      �?       @       @      @              @      @      @      @      @      @      $@      (@       @      @      ,@      @      C@      4@      ^@      7@      8@      8@      >@      =@      A@     @R@     �D@      H@      A@     �A@      I@     �W@     @Q@      K@     @V@      [@      S@      `@     �b@      [@     @d@     @c@     �e@     �r@     �i@     `y@     �t@     �s@     �v@     `v@      ~@     �@     �z@     Ё@     x�@     ��@     P�@     ��@     �@     `�@     ؏@     h�@     �@     X�@     d�@     �@     ̞@     ��@     ��@     ��@     b�@     ��@     ��@     ��@     H�@     ί@     "�@     ?�@     ݶ@     k�@     Z�@     >�@     7�@    �/�@    ��@     $�@     k�@    �~�@    ���@     ��@     J�@     ��@     ��@    ��@     ��@     ��@    ���@    ���@    �L�@     ξ@     N�@     �@     @�@     r�@     �@     ��@     ��@     �@      �@      x@      h@     �P@      =@       @        
�
Conv2/h_conv2*�   @S�?     $#A! D��#�@)y�yH89�@2�	        �-���q=.��fc��>39W$:��>���?�ګ>����>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�	            �A              �?              �?              �?              �?      �?              �?      �?      �?       @              �?      �?       @              �?              �?       @       @              @       @      @      B@      �?      �?      �?       @       @      @              @      @      @      @      @      @      $@      (@       @      @      ,@      @      C@      4@      ^@      7@      8@      8@      >@      =@      A@     @R@     �D@      H@      A@     �A@      I@     �W@     @Q@      K@     @V@      [@      S@      `@     �b@      [@     @d@     @c@     �e@     �r@     �i@     `y@     �t@     �s@     �v@     `v@      ~@     �@     �z@     Ё@     x�@     ��@     P�@     ��@     �@     `�@     ؏@     h�@     �@     X�@     d�@     �@     ̞@     ��@     ��@     ��@     b�@     ��@     ��@     ��@     H�@     ί@     "�@     ?�@     ݶ@     k�@     Z�@     >�@     7�@    �/�@    ��@     $�@     k�@    �~�@    ���@     ��@     J�@     ��@     ��@    ��@     ��@     ��@    ���@    ���@    �L�@     ξ@     N�@     �@     @�@     r�@     �@     ��@     ��@     �@      �@      x@      h@     �P@      =@       @        

cross_entropy_scl�с=

training_accuracyH�z?eq~y�S      $_�	?8����A�*��
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   pIDAT(�͐��0-�� Y�A؃E
s� �0? E�@��ɧ4���1I�CS@&����P4[�&!�QZ{.��G��QI"MZ�x��u�,�/v�Z�_����4���������"l@]    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   �IDAT(�c`� q���y��'��10000T�=�$��L�xQ�@%�28A�T0$�nC�}�����?��m1tn��?��������^�ν��*����,Pzy�X��3��rN�/������_db6�������A��e``�J2C���w�#��g^��tXҿ�f�^���lޘZ`���u8u2af�)y�/Nɽ�~��e�`�� �u�8%� JY/��}    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H   �IDAT(�c`���ƈS���-��J��(F�1\��Sg��r��I�����w�.I���8MU�1U�S�e�����_���T��S���������aHf����~����wc�z������J|��&)���Ǫ��L�fB�t���!��3)����эM�^���^�77�1%'���������i.��E��L�{0|¹�������_�%�V{�i�'3�� ��V����    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(��бKa��^��.�$�9f�C��DKA��cJK��CBC��9D�D��D���x���#����y>���<��d�$��^7�fg�pW��|��l�|DpR	�6w_��=�Y�G<�m�`�g.; ���}�ڵMY��Y�m��=�w�0g7}vJ�XxH��v�W寥�9��7q���±��ռ43���q���7�q:$�Z7O�����$I��U\d�?Z�h�F��X    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(�c`���` nɩ�>z"xL���sK�����[�:u����NTp�ncDV0�ơ1�ϟ��Dq�g``�������A� [������m��G@�x$o,�#��>c�H��V(=�1Q��YB5  J	*L�G�    IEND�B`�
%
Conv1/weights/summaries/mean_1��E�
'
 Conv1/weights/summaries/stddev_1�>�=
$
Conv1/weights/summaries/max_1�K>
$
Conv1/weights/summaries/min_1�oX�
�
!Conv1/weights/summaries/histogram*�	   ��˿   �bp�?      �@!  )��I�)�E���@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ���Zr[v�>O�ʗ��>�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @      "@      &@      0@      3@      5@      8@      2@      .@      6@      8@      *@      .@      0@      .@      *@      1@      (@      2@      .@      *@      &@      @      @      "@      @      "@       @      @      @      @              �?      �?      @       @       @               @      @      @              �?      �?              �?      �?               @              �?              �?      �?              �?              �?              @       @      �?       @               @               @      �?      �?      �?       @      @      �?      @      �?      @      @      $@               @       @      @      @      "@      $@      (@      @      @       @      ,@      .@      1@      ,@      3@      3@      6@      $@      ,@      4@      &@      7@      0@      (@      2@      (@        
$
Conv1/biases/summaries/mean_1�<�=
&
Conv1/biases/summaries/stddev_1���;
#
Conv1/biases/summaries/max_17�=
#
Conv1/biases/summaries/min_1��=
�
 Conv1/biases/summaries/histogram*�	   ��#�?    �&�?      @@!   X��@)�u	Ym�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              "@      1@      @        
�"
Conv1/conv1_wx_b*�!	   ��w��    pk�?     $3A! ���@)-d2�Z�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;39W$:��>R%�����>�5�L�>;9��R�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              ,@     pq@     8�@     `�@      �@     ܘ@     ��@     ڣ@     0�@     j�@     �@     ȭ@     ��@     ձ@     ��@     w�@      �@     ��@     ɴ@     |�@     (�@     ��@     ɵ@     q�@     ��@     �@     r�@     C�@     ��@     N�@     ��@     P�@     8�@     j�@     �@     �@     ��@     ,�@     L�@     L�@     ��@     �@     Ж@     �@      �@     <�@     �@     H�@     ��@     P�@     ��@     ��@     H�@     ��@     �}@     0{@     `z@     �v@     �u@     �s@     Pq@     �p@     �p@     `m@     �h@     �f@     �g@     �d@     �b@      `@      \@      W@     �X@      V@     �S@     @S@     �J@     �K@      M@      I@      L@     �D@     �C@     �@@      8@      9@      <@      ;@      ;@      *@      1@      2@      ,@      4@      "@      $@      "@      @       @      @      @      @      @      @      @      @      @      @               @      @       @       @       @      �?      @      @      �?               @      �?      �?       @              �?      �?              �?              �?              �?              �?              �?               @      �?       @              �?               @               @              �?              @      @              �?      �?      @       @      �?      @      @              @      @      @      @      @      @      $@       @       @      &@      (@      1@      $@      &@      $@      5@      .@      ;@      <@      ?@      =@     �A@     �A@      <@     �C@      F@     �F@     �P@      Q@     �O@      T@      S@     @S@      T@     �Z@      `@      b@     �b@      c@      h@      h@     @i@     �n@     �n@     �s@     �q@     �t@      x@     �y@      |@     ��@     ��@     �@     ��@     �@     ��@     X�@     L�@     ��@     ؒ@     x�@     ��@     ��@     �@     Ơ@     �@     ��@     ֧@     �@     n�@     ��@     P�@     �@     �@    ��@    ���@     ��@    8�
A    ��A    �A     ��@     ��@     ��@    ���@     J�@     :�@    ���@     I�@     ��@     ��@    �(�@    �B�@    �]�@    ���@     A�@     ��@     1�@     ȱ@     v�@     Υ@     ��@     $�@     P�@     �d@     �C@      @        
�
Conv1/h_conv1*�    pk�?     $3A!���ˊA)��j���@2�        �-���q=39W$:��>R%�����>�5�L�>;9��R�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            ��A              �?              �?               @      �?       @              �?               @               @              �?              @      @              �?      �?      @       @      �?      @      @              @      @      @      @      @      @      $@       @       @      &@      (@      1@      $@      &@      $@      5@      .@      ;@      <@      ?@      =@     �A@     �A@      <@     �C@      F@     �F@     �P@      Q@     �O@      T@      S@     @S@      T@     �Z@      `@      b@     �b@      c@      h@      h@     @i@     �n@     �n@     �s@     �q@     �t@      x@     �y@      |@     ��@     ��@     �@     ��@     �@     ��@     X�@     L�@     ��@     ؒ@     x�@     ��@     ��@     �@     Ơ@     �@     ��@     ֧@     �@     n�@     ��@     P�@     �@     �@    ��@    ���@     ��@    8�
A    ��A    �A     ��@     ��@     ��@    ���@     J�@     :�@    ���@     I�@     ��@     ��@    �(�@    �B�@    �]�@    ���@     A�@     ��@     1�@     ȱ@     v�@     Υ@     ��@     $�@     P�@     �d@     �C@      @        
%
Conv2/weights/summaries/mean_1
a��
'
 Conv2/weights/summaries/stddev_1%-�=
$
Conv2/weights/summaries/max_1�U>
$
Conv2/weights/summaries/min_1xn]�
�
!Conv2/weights/summaries/histogram*�	    ϭ˿    ܰ�?      �@! ���y�R�)I�C�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(��u`P+d�>0�6�/n�>
�/eq
�>;�"�q�>�uE����>�f����>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              U@     �@     ��@     (�@     l�@     đ@     ��@     ��@     ̓@     �@     P�@     ��@     ��@     �@     ��@     X�@     ��@     ��@      �@     X�@     Ђ@     ��@     �@     p|@     �x@     �w@     v@     `t@     �q@     �p@     @p@     �j@      j@     `f@     �f@     @b@     @a@     �[@     �V@     �Y@      W@     �V@     @U@     @R@     �O@     �K@     �M@     �M@      D@      C@      C@      9@      @@      ;@      9@      5@      ;@      ,@      0@      0@      0@      0@      1@      $@       @      @      *@      (@      @      @      @      @      @      @      @      @      �?      @      �?      �?      �?      �?      @      @      @               @              �?      �?       @      �?              �?      �?               @              �?              �?              �?              �?              �?       @      �?               @              �?       @      @       @              @      @      @      @              @      @       @       @      @      @      @      @      �?      @      "@      @      @      "@      ,@      ,@      @      2@      4@      (@      0@      4@      7@      =@      ?@     �@@     �C@      F@      H@      D@      D@      F@      L@      Q@      Q@     �U@      V@     @R@     @Z@     �]@     �_@     �a@     �c@      c@     �h@     �i@     p@     �l@     Pq@     pp@     `s@     v@     pz@     z@     �|@     P}@     0�@     ��@     0�@     ��@     (�@      �@     ��@      �@     ؎@     �@     �@     D�@     ܒ@     \�@     (�@     h�@     ȑ@     ��@     0�@     0�@     �~@      D@        
$
Conv2/biases/summaries/mean_1��=
&
Conv2/biases/summaries/stddev_1��Z;
#
Conv2/biases/summaries/max_1T�=
#
Conv2/biases/summaries/min_1�3�=
�
 Conv2/biases/summaries/histogram*�	   @x��?   �
�?      P@!  �@ �@)�rXXy��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @      M@      @        
�$
Conv2/conv2_wx_b*�$	   �F,�   �s��?     $#A! �tG����)�pN��r�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�[�=�k���*��ڽ�G&�$���4[_>������m!#���u��gr�>�MZ��K�>���]���>�5�L�>G&�$�>�*��ڽ>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              �?      *@      F@     @]@     `r@     Ѐ@     �@     8�@     ��@     ئ@     ��@     �@     n�@     �@     8�@     ׾@    �!�@    ��@     E�@     ��@    ���@    ��@    �`�@    ���@    ���@     ��@     X�@     9�@     ��@    �<�@     +�@     D�@     ��@    �N�@     ��@     e�@     v�@     ��@     i�@     ��@     �@     &�@     ޫ@     p�@     �@     ��@     �@     &�@     ��@     N�@     О@     �@     t�@     ��@     ؗ@     D�@     ��@     ��@      �@     ��@     ��@     ؂@     d�@     P�@     p�@     �|@     z@     �u@     `�@     Pp@      j@     �g@      l@     �q@     �a@      q@     �c@     �z@     @g@      h@     �a@     �Q@      V@     �Y@     �M@     �G@     �N@     �F@      E@      B@      J@     �@@      >@      <@      9@      0@      ,@      6@      .@      .@      .@      &@      0@      (@       @      @      H@      @      "@      @       @      @       @      �?      @       @      $@      @       @      @      @       @       @              @      �?       @       @      �?              �?              �?              �?              �?               @              �?      �?              �?              *@               @              �?               @              �?      �?              �?       @      �?              �?       @       @       @              �?              @       @      �?      �?      �?              �?       @      @      @      @      @      @      @      @      @      @      @      @      &@      &@      *@      "@      @      2@      .@      2@      6@      7@      5@      @@      >@      >@      A@     �C@      C@     �C@     �O@     @n@     @Q@     �P@     @Q@     �P@      T@     @`@      Z@     �a@     �`@      i@     �a@     �b@      g@     @m@     �r@     `l@     �x@     �y@     �t@     �y@     H�@     ��@     �@     (�@     ��@     ؍@     Ў@     P�@     ��@     ��@     ؒ@     $�@      �@     ��@     ě@     $�@     ڡ@     ��@     f�@     p�@     ȧ@     ��@     �@     ��@     N�@     V�@     �@     �@     �@     F�@     >�@     s�@    �^�@     ��@     ��@     ,�@    �x�@     6�@     ��@     ^�@     ��@     ��@    �!�@     ��@     ��@    ���@     ��@     ��@     F�@     .�@     D�@     W�@     ��@     �@     h�@     ��@     ��@     �x@     `e@      S@      5@      @        
�
Conv2/h_conv2*�   �s��?     $#A! ���GC�@)j�B[��@2�	        �-���q=�u��gr�>�MZ��K�>���]���>�5�L�>G&�$�>�*��ڽ>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�	            L�A              *@               @              �?               @              �?      �?              �?       @      �?              �?       @       @       @              �?              @       @      �?      �?      �?              �?       @      @      @      @      @      @      @      @      @      @      @      @      &@      &@      *@      "@      @      2@      .@      2@      6@      7@      5@      @@      >@      >@      A@     �C@      C@     �C@     �O@     @n@     @Q@     �P@     @Q@     �P@      T@     @`@      Z@     �a@     �`@      i@     �a@     �b@      g@     @m@     �r@     `l@     �x@     �y@     �t@     �y@     H�@     ��@     �@     (�@     ��@     ؍@     Ў@     P�@     ��@     ��@     ؒ@     $�@      �@     ��@     ě@     $�@     ڡ@     ��@     f�@     p�@     ȧ@     ��@     �@     ��@     N�@     V�@     �@     �@     �@     F�@     >�@     s�@    �^�@     ��@     ��@     ,�@    �x�@     6�@     ��@     ^�@     ��@     ��@    �!�@     ��@     ��@    ���@     ��@     ��@     F�@     .�@     D�@     W�@     ��@     �@     h�@     ��@     ��@     �x@     `e@      S@      5@      @        

cross_entropy_scl��>

training_accuracyH�z?�H��T      5bA	��ق���A�*ѩ
�
Input_Reshape/input_img/image/0"�"��PNG

   IHDR          Wf�H   �IDAT(�͑�KBQ�O�4p
ũ�	)�H��DSKE��-����� ����P�FP�4(BѦ}���������垏�4�Y�T��������@�b�քǬW-x�%I�{�&b�'3/k w)I��Z20"�yݓ4��c`�]S&?��E6^>�c�NU8���(��yvCi��C(T���3�x�R���|�3���JU��_��p�Z� <�ʎ�(� ����C�����X୶nu��޴���/�\j$]�4�    IEND�B`�
�
Input_Reshape/input_img/image/1"�"��PNG

   IHDR          Wf�H   IDAT(�c`h ��.��[oc��}T�H1�}��Q���U,�r��200��}�)g{�o�������>S```X���	Sr��V?�95�o+�֍�"�$��t�?����˿�>"� Y���:��.vIq��]�k�6I`1�����i/�1%����0000$�]�a�N�� LI5���<�`��w!纯��0$����������NvL��}�Ps���la�����g5�H�  ��SD�Iu    IEND�B`�
�
Input_Reshape/input_img/image/2"�"��PNG

   IHDR          Wf�H  IDAT(��ѡKa�gbҰ��d�`�[���-� ��ei���e�"��,8�V�2.�n�W��<��N�w_4���������*y�k\�vq��IRdX�ԭ=�}� [8�- XL���WX�/�m��}�e9 z(���������ӏ^�f�g$��2S�%���K��~�H~l�4 ��
ꑬ�q/sV����X�v� �|β��� ���J�).9@���)��4���ӯ������P��k=�'I+���LR��Ec_$u�k���o#á���o    IEND�B`�
�
Input_Reshape/input_img/image/3"�"��PNG

   IHDR          Wf�H   �IDAT(�c`��=t��̇��#�)ܒ��'q��t�(Nɦ�8�w�`�)�� ����l8%���s؎SN��wo�����20H��>�'�)��߄���?��CN�s����
̽o1$��˺~B��Ar��E�r,7��z��.���f�����h¹�+4�8��������&��`8�_�:V�! �VC��̍    IEND�B`�
�
Input_Reshape/input_img/image/4"�"��PNG

   IHDR          Wf�H   �IDAT(��н
�Q���[d�6�B"��M	��.�� ��f��5Ȫd��Ky�� t�g��tz>�y���M����j[�:��a���sK���%v���.X?0�ax5m�-��f�&p�ip�5��:�U���GH.��/��!�$���W��r�@E�"F����	a�qϩ���~Ak����궅��:!�DZ`Z��ٟ�M�dB��-    IEND�B`�
%
Conv1/weights/summaries/mean_1sF�
'
 Conv1/weights/summaries/stddev_1�s�=
$
Conv1/weights/summaries/max_15!L>
$
Conv1/weights/summaries/min_1�Z�
�
!Conv1/weights/summaries/histogram*�	    �A˿   �&��?      �@! ���=V�)���x�@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`���bB�SY�ܗ�SsW�<DKc��T��lDZrS�uܬ�@8���%�V6�U�4@@�$?+A�F�&?
����G?�qU���I?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�              @      "@      &@      .@      6@      2@      :@      2@      ,@      3@      <@      &@      0@      ,@      0@      *@      1@      *@      2@      .@      (@      &@      @      @      @       @      @      "@      @      @      @       @      �?      @              @      �?               @      @       @       @      �?      �?              �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?               @               @      �?              @               @      �?       @       @      @      �?       @      @      @      @      $@       @      @      @      @      @       @      $@      ,@      @      @      $@      *@      *@      0@      .@      5@      2@      6@      $@      .@      2@      &@      8@      .@      *@      2@      (@        
$
Conv1/biases/summaries/mean_1���=
&
Conv1/biases/summaries/stddev_1?e�;
#
Conv1/biases/summaries/max_1�J�=
#
Conv1/biases/summaries/min_1��=
�
 Conv1/biases/summaries/histogram*�	   ����?    T)�?      @@!   V��@)��?�]a�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      3@      @        
�"
Conv1/conv1_wx_b*�"	    x���   ����?     $3A! ImЃ��@)�vs�c�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ['�?�;;�"�qʾG&�$��5�"�g����u`P+d�>0�6�/n�>��~���>�XQ��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              "@     @m@     (�@     ��@     �@     H�@     ��@     (�@     ��@     p�@     Ҫ@     !�@     Z�@     ��@     ��@     ��@     Q�@     ش@     ��@     ��@     ��@     ��@     w�@     µ@     ��@     ��@     ��@     �@     ��@     ֯@     @�@     2�@     Ϊ@     ��@     R�@     ��@     6�@     R�@     4�@     �@     ,�@     �@     �@     �@     h�@     ��@     (�@      �@     �@     H�@     �@     ��@     ��@     ��@      �@     �{@      {@      y@     `v@      v@     �t@      p@     �m@      m@     �j@     �g@     `f@     `b@      b@     @^@     �^@      ]@     �Y@      R@      T@      T@      R@     @Q@     �M@      @@      G@      E@      C@      <@      A@      4@     �@@      0@      4@      1@      &@      $@      "@      $@      ,@      ,@      @      @      @      @      @      @      @      @      @      @      @      @      @      �?      �?       @       @      �?      �?              �?              �?      �?               @              �?              �?               @              �?              �?              �?              @              �?              �?      �?      �?       @      �?              �?      �?               @       @       @      �?              �?       @       @       @      @      �?      @      @      @       @      @       @       @      @      @       @      ,@      &@      &@      0@      "@      0@      $@      (@      5@      ?@      8@      2@      :@      9@     �D@      D@      D@      H@     �I@     �I@     �P@     �L@      Q@     �T@     �X@     �W@     �X@     �[@     �a@     �d@     @d@      e@     �g@     �i@     `n@     �p@     �r@     �r@     �t@     �w@     |@     P}@     `~@      �@     P�@     P�@      �@     �@     ��@     �@     �@     ԓ@     `�@     �@     ��@     .�@     ơ@     P�@     �@     ��@     D�@     �@     h�@     �@     ķ@     ׼@    ���@    ���@    ���@    X�A    P�A    �_A     .�@     ��@     ��@     ��@    ���@     z�@     ��@     �@     ��@     R�@    �w�@    ���@     ��@     ��@     ݿ@     V�@     ��@     ,�@     ��@     �@     �@     P�@     ��@     �d@     �B@      @        
�
Conv1/h_conv1*�   ����?     $3A!���T�A)�~~ӳ�@2�        �-���q=�u`P+d�>0�6�/n�>��~���>�XQ��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�            x!A              �?              �?              @              �?              �?      �?      �?       @      �?              �?      �?               @       @       @      �?              �?       @       @       @      @      �?      @      @      @       @      @       @       @      @      @       @      ,@      &@      &@      0@      "@      0@      $@      (@      5@      ?@      8@      2@      :@      9@     �D@      D@      D@      H@     �I@     �I@     �P@     �L@      Q@     �T@     �X@     �W@     �X@     �[@     �a@     �d@     @d@      e@     �g@     �i@     `n@     �p@     �r@     �r@     �t@     �w@     |@     P}@     `~@      �@     P�@     P�@      �@     �@     ��@     �@     �@     ԓ@     `�@     �@     ��@     .�@     ơ@     P�@     �@     ��@     D�@     �@     h�@     �@     ķ@     ׼@    ���@    ���@    ���@    X�A    P�A    �_A     .�@     ��@     ��@     ��@    ���@     z�@     ��@     �@     ��@     R�@    �w�@    ���@     ��@     ��@     ݿ@     V�@     ��@     ,�@     ��@     �@     �@     P�@     ��@     �d@     �B@      @        
%
Conv2/weights/summaries/mean_1Mp��
'
 Conv2/weights/summaries/stddev_1G1�=
$
Conv2/weights/summaries/max_1��U>
$
Conv2/weights/summaries/min_1�p^�
�
!Conv2/weights/summaries/histogram*�	   ��˿   ࿶�?      �@!�����R�)]:�/o�x@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[���FF�G �>�?�s����h���`�8K�ߝ���~��¾�[�=�k����~]�[�>��>M|K�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �U@     ��@     ��@     P�@     \�@     đ@     �@     �@     ��@      �@     8�@     `�@     Ȑ@     ��@     ؍@     ��@     �@     �@     h�@     ��@     x�@     ��@     �@      {@     �x@     �x@     pv@     `t@     �q@     �p@      o@      l@     �h@     �h@     �c@     �c@     �`@     �^@      Z@      V@      W@     �V@     @T@     @U@     �K@      L@      N@      E@     �E@      G@      C@      <@     �A@      9@      4@      7@      3@      0@      2@      3@      1@      $@      (@      (@      ,@      ,@      "@      @      @      @      @      @      @      @      @      @      &@      @      @               @      @      @      @      �?      �?      �?              �?              @              �?              �?              �?               @      �?              �?              �?      �?      �?       @              �?              �?      �?      @      @      @      @               @      @      @      @      @      @      @       @       @      &@       @       @      @      @      &@      *@      *@      4@      2@      3@      ,@      2@      ;@      9@      8@      @@      >@     �E@     �B@     �H@      J@     �L@      D@      O@      Q@     �U@     �V@      S@     @[@      Z@     �`@      d@      c@     �b@      h@     �k@     �m@     `k@      q@     Pr@     �r@     �v@     �y@      {@     �{@     `}@     ��@     X�@     Є@     ��@     �@     ��@     p�@     Ќ@     �@     �@     �@     \�@     ��@     <�@     8�@     ��@     ��@     ��@     (�@     �@      @      D@        
$
Conv2/biases/summaries/mean_1 ��=
&
Conv2/biases/summaries/stddev_1��Y;
#
Conv2/biases/summaries/max_1(�=
#
Conv2/biases/summaries/min_1|��=
�
 Conv2/biases/summaries/histogram*�	   ��s�?    ���?      P@!   ���@)��-��}�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      L@      @        
�%
Conv2/conv2_wx_b*�$	   `/��   �Ī�?     $#A! 0���)�� u���@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|KվK+�E��Ͼ['�?�;;�"�qʾ��~��¾�[�=�k����������?�ګ��i����v>E'�/��x>K���7�>u��6
�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�XQ��>�����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              �?       @      $@      S@     @c@     pr@     ��@     ��@     З@     �@     
�@     p�@     5�@     �@     ��@     ��@     ��@    �e�@    ���@     :�@     ��@     *�@    �u�@     �@     ��@    ��@    ���@    ���@     ��@     }�@    ���@     6�@    �/�@     ߾@     ��@     g�@     ��@     �@     �@     Ƕ@     ߲@     ��@     �@     ʯ@     ��@     4�@     �@     T�@     ʦ@     �@     �@     ��@     ܞ@     ��@     <�@     X�@     �@     ��@     �@     ��@     @�@     ��@      �@      �@     �@     �{@     �@     �~@     @�@     �o@     �p@     �v@     �m@      h@      i@     �d@      c@     �l@     �e@     @]@      `@     @V@     �]@      T@      J@     �J@     �K@      I@      N@      >@     �C@     �@@     �M@      @@      ?@      9@      :@      3@      2@      I@      ,@      0@      @      &@      2@      &@       @      (@      "@      @      @      @      @      "@       @      @      �?               @       @              @      @      �?              �?               @      �?       @              @      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?      �?      �?              �?      �?      �?       @      �?       @       @               @      �?      @      @      @      "@      @      R@      @      (@      @      "@      (@      ,@      1@      &@      (@      6@      5@      =@      ;@      <@      <@      >@      J@      B@     @R@     �H@     �I@     �K@     �R@      X@     @Y@     �Z@     �V@      [@     �`@     �d@     �a@     �d@      f@     �j@     @j@     ~@     �q@     �v@     pr@     `{@     �v@      y@     p�@     �~@     ��@     �@     (�@     ��@     X�@     (�@     ܒ@     \�@     ��@     ��@     h�@     X�@     �@     J�@     Ơ@     ��@     ��@     d�@     ��@     *�@     �@     r�@     ��@     �@     ��@     A�@     ��@     %�@     '�@     g�@    �)�@    ��@    ���@     ?�@    ��@    ���@    �1�@     ��@    ��@    ���@     +�@     ��@     �@    ���@    �%�@     �@     Ż@     ��@     m�@     °@     Ϊ@     ��@     �@     В@     X�@      y@     �j@      S@     �A@      @      �?        
�
Conv2/h_conv2*�   �Ī�?     $#A! \fϦ=�@)ȁ% ��@2�	        �-���q=�i����v>E'�/��x>K���7�>u��6
�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�XQ��>�����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�	            L�A              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?      �?      �?              �?      �?      �?       @      �?       @       @               @      �?      @      @      @      "@      @      R@      @      (@      @      "@      (@      ,@      1@      &@      (@      6@      5@      =@      ;@      <@      <@      >@      J@      B@     @R@     �H@     �I@     �K@     �R@      X@     @Y@     �Z@     �V@      [@     �`@     �d@     �a@     �d@      f@     �j@     @j@     ~@     �q@     �v@     pr@     `{@     �v@      y@     p�@     �~@     ��@     �@     (�@     ��@     X�@     (�@     ܒ@     \�@     ��@     ��@     h�@     X�@     �@     J�@     Ơ@     ��@     ��@     d�@     ��@     *�@     �@     r�@     ��@     �@     ��@     A�@     ��@     %�@     '�@     g�@    �)�@    ��@    ���@     ?�@    ��@    ���@    �1�@     ��@    ��@    ���@     +�@     ��@     �@    ���@    �%�@     �@     Ż@     ��@     m�@     °@     Ϊ@     ��@     �@     В@     X�@      y@     �j@      S@     �A@      @      �?        

cross_entropy_scl�&�=

training_accuracyH�z?��*�