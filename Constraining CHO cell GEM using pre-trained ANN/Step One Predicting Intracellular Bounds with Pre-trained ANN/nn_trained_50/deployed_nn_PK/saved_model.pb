��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8��
~
dense_3358/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*"
shared_namedense_3358/kernel
w
%dense_3358/kernel/Read/ReadVariableOpReadVariableOpdense_3358/kernel*
_output_shapes

:2*
dtype0
v
dense_3358/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_namedense_3358/bias
o
#dense_3358/bias/Read/ReadVariableOpReadVariableOpdense_3358/bias*
_output_shapes
:2*
dtype0
~
dense_3359/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*"
shared_namedense_3359/kernel
w
%dense_3359/kernel/Read/ReadVariableOpReadVariableOpdense_3359/kernel*
_output_shapes

:2*
dtype0
v
dense_3359/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3359/bias
o
#dense_3359/bias/Read/ReadVariableOpReadVariableOpdense_3359/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_3358/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*)
shared_nameAdam/dense_3358/kernel/m
�
,Adam/dense_3358/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3358/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/dense_3358/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/dense_3358/bias/m
}
*Adam/dense_3358/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3358/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_3359/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*)
shared_nameAdam/dense_3359/kernel/m
�
,Adam/dense_3359/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3359/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/dense_3359/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3359/bias/m
}
*Adam/dense_3359/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3359/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_3358/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*)
shared_nameAdam/dense_3358/kernel/v
�
,Adam/dense_3358/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3358/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/dense_3358/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/dense_3358/bias/v
}
*Adam/dense_3358/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3358/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_3359/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*)
shared_nameAdam/dense_3359/kernel/v
�
,Adam/dense_3359/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3359/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/dense_3359/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3359/bias/v
}
*Adam/dense_3359/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3359/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�
iter

beta_1

beta_2
	decay
learning_rate	m4
m5m6m7	v8
v9v:v;

	0

1
2
3

	0

1
2
3
 
�
trainable_variables
metrics
layer_metrics
	variables
regularization_losses

layers
layer_regularization_losses
non_trainable_variables
 
][
VARIABLE_VALUEdense_3358/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3358/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
�
trainable_variables
metrics
 layer_metrics
	variables
regularization_losses

!layers
"layer_regularization_losses
#non_trainable_variables
][
VARIABLE_VALUEdense_3359/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3359/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
$metrics
%layer_metrics
	variables
regularization_losses

&layers
'layer_regularization_losses
(non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
4
	+total
	,count
-	variables
.	keras_api
D
	/total
	0count
1
_fn_kwargs
2	variables
3	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

-	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

2	variables
�~
VARIABLE_VALUEAdam/dense_3358/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3358/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_3359/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3359/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_3358/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3358/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_3359/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3359/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_input_1230Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1230dense_3358/kerneldense_3358/biasdense_3359/kerneldense_3359/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *.
f)R'
%__inference_signature_wrapper_6824773
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_3358/kernel/Read/ReadVariableOp#dense_3358/bias/Read/ReadVariableOp%dense_3359/kernel/Read/ReadVariableOp#dense_3359/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/dense_3358/kernel/m/Read/ReadVariableOp*Adam/dense_3358/bias/m/Read/ReadVariableOp,Adam/dense_3359/kernel/m/Read/ReadVariableOp*Adam/dense_3359/bias/m/Read/ReadVariableOp,Adam/dense_3358/kernel/v/Read/ReadVariableOp*Adam/dense_3358/bias/v/Read/ReadVariableOp,Adam/dense_3359/kernel/v/Read/ReadVariableOp*Adam/dense_3359/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8� *)
f$R"
 __inference__traced_save_6825028
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3358/kerneldense_3358/biasdense_3359/kerneldense_3359/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_3358/kernel/mAdam/dense_3358/bias/mAdam/dense_3359/kernel/mAdam/dense_3359/bias/mAdam/dense_3358/kernel/vAdam/dense_3358/bias/vAdam/dense_3359/kernel/vAdam/dense_3359/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8� *,
f'R%
#__inference__traced_restore_6825101Ɲ
�
�
1__inference_sequential_1229_layer_call_fn_6824738

input_1230
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_1230unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *U
fPRN
L__inference_sequential_1229_layer_call_and_return_conditional_losses_68247272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
input_1230
�
�
G__inference_dense_3358_layer_call_and_return_conditional_losses_6824572

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Tanh�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^dense_3358/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_3359_layer_call_fn_6824920

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3359_layer_call_and_return_conditional_losses_68246042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�!
�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824659

input_1230
dense_3358_6824636
dense_3358_6824638
dense_3359_6824641
dense_3359_6824643
identity��"dense_3358/StatefulPartitionedCall�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�"dense_3359/StatefulPartitionedCall�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
"dense_3358/StatefulPartitionedCallStatefulPartitionedCall
input_1230dense_3358_6824636dense_3358_6824638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3358_layer_call_and_return_conditional_losses_68245722$
"dense_3358/StatefulPartitionedCall�
"dense_3359/StatefulPartitionedCallStatefulPartitionedCall+dense_3358/StatefulPartitionedCall:output:0dense_3359_6824641dense_3359_6824643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3359_layer_call_and_return_conditional_losses_68246042$
"dense_3359/StatefulPartitionedCall�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3358_6824636*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3359_6824641*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentity+dense_3359/StatefulPartitionedCall:output:0#^dense_3358/StatefulPartitionedCall4^dense_3358/kernel/Regularizer/Square/ReadVariableOp#^dense_3359/StatefulPartitionedCall4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2H
"dense_3358/StatefulPartitionedCall"dense_3358/StatefulPartitionedCall2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp2H
"dense_3359/StatefulPartitionedCall"dense_3359/StatefulPartitionedCall2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:���������
$
_user_specified_name
input_1230
�
�
1__inference_sequential_1229_layer_call_fn_6824844

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *U
fPRN
L__inference_sequential_1229_layer_call_and_return_conditional_losses_68246882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_6824942@
<dense_3359_kernel_regularizer_square_readvariableop_resource
identity��3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<dense_3359_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentity%dense_3359/kernel/Regularizer/mul:z:04^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp
�
�
G__inference_dense_3358_layer_call_and_return_conditional_losses_6824880

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Tanh�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^dense_3358/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_dense_3359_layer_call_and_return_conditional_losses_6824604

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�!
�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824633

input_1230
dense_3358_6824583
dense_3358_6824585
dense_3359_6824615
dense_3359_6824617
identity��"dense_3358/StatefulPartitionedCall�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�"dense_3359/StatefulPartitionedCall�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
"dense_3358/StatefulPartitionedCallStatefulPartitionedCall
input_1230dense_3358_6824583dense_3358_6824585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3358_layer_call_and_return_conditional_losses_68245722$
"dense_3358/StatefulPartitionedCall�
"dense_3359/StatefulPartitionedCallStatefulPartitionedCall+dense_3358/StatefulPartitionedCall:output:0dense_3359_6824615dense_3359_6824617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3359_layer_call_and_return_conditional_losses_68246042$
"dense_3359/StatefulPartitionedCall�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3358_6824583*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3359_6824615*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentity+dense_3359/StatefulPartitionedCall:output:0#^dense_3358/StatefulPartitionedCall4^dense_3358/kernel/Regularizer/Square/ReadVariableOp#^dense_3359/StatefulPartitionedCall4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2H
"dense_3358/StatefulPartitionedCall"dense_3358/StatefulPartitionedCall2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp2H
"dense_3359/StatefulPartitionedCall"dense_3359/StatefulPartitionedCall2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:���������
$
_user_specified_name
input_1230
�
�
%__inference_signature_wrapper_6824773

input_1230
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_1230unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *+
f&R$
"__inference__wrapped_model_68245512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
input_1230
�
�
1__inference_sequential_1229_layer_call_fn_6824699

input_1230
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_1230unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *U
fPRN
L__inference_sequential_1229_layer_call_and_return_conditional_losses_68246882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
input_1230
�
�
G__inference_dense_3359_layer_call_and_return_conditional_losses_6824911

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_6824931@
<dense_3358_kernel_regularizer_square_readvariableop_resource
identity��3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<dense_3358_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
IdentityIdentity%dense_3358/kernel/Regularizer/mul:z:04^dense_3358/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp
�3
�
 __inference__traced_save_6825028
file_prefix0
,savev2_dense_3358_kernel_read_readvariableop.
*savev2_dense_3358_bias_read_readvariableop0
,savev2_dense_3359_kernel_read_readvariableop.
*savev2_dense_3359_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_dense_3358_kernel_m_read_readvariableop5
1savev2_adam_dense_3358_bias_m_read_readvariableop7
3savev2_adam_dense_3359_kernel_m_read_readvariableop5
1savev2_adam_dense_3359_bias_m_read_readvariableop7
3savev2_adam_dense_3358_kernel_v_read_readvariableop5
1savev2_adam_dense_3358_bias_v_read_readvariableop7
3savev2_adam_dense_3359_kernel_v_read_readvariableop5
1savev2_adam_dense_3359_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_3358_kernel_read_readvariableop*savev2_dense_3358_bias_read_readvariableop,savev2_dense_3359_kernel_read_readvariableop*savev2_dense_3359_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_dense_3358_kernel_m_read_readvariableop1savev2_adam_dense_3358_bias_m_read_readvariableop3savev2_adam_dense_3359_kernel_m_read_readvariableop1savev2_adam_dense_3359_bias_m_read_readvariableop3savev2_adam_dense_3358_kernel_v_read_readvariableop1savev2_adam_dense_3358_bias_v_read_readvariableop3savev2_adam_dense_3359_kernel_v_read_readvariableop1savev2_adam_dense_3359_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapesx
v: :2:2:2:: : : : : : : : : :2:2:2::2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
�
�
"__inference__wrapped_model_6824551

input_1230=
9sequential_1229_dense_3358_matmul_readvariableop_resource>
:sequential_1229_dense_3358_biasadd_readvariableop_resource=
9sequential_1229_dense_3359_matmul_readvariableop_resource>
:sequential_1229_dense_3359_biasadd_readvariableop_resource
identity��1sequential_1229/dense_3358/BiasAdd/ReadVariableOp�0sequential_1229/dense_3358/MatMul/ReadVariableOp�1sequential_1229/dense_3359/BiasAdd/ReadVariableOp�0sequential_1229/dense_3359/MatMul/ReadVariableOp�
0sequential_1229/dense_3358/MatMul/ReadVariableOpReadVariableOp9sequential_1229_dense_3358_matmul_readvariableop_resource*
_output_shapes

:2*
dtype022
0sequential_1229/dense_3358/MatMul/ReadVariableOp�
!sequential_1229/dense_3358/MatMulMatMul
input_12308sequential_1229/dense_3358/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22#
!sequential_1229/dense_3358/MatMul�
1sequential_1229/dense_3358/BiasAdd/ReadVariableOpReadVariableOp:sequential_1229_dense_3358_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1sequential_1229/dense_3358/BiasAdd/ReadVariableOp�
"sequential_1229/dense_3358/BiasAddBiasAdd+sequential_1229/dense_3358/MatMul:product:09sequential_1229/dense_3358/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22$
"sequential_1229/dense_3358/BiasAdd�
sequential_1229/dense_3358/TanhTanh+sequential_1229/dense_3358/BiasAdd:output:0*
T0*'
_output_shapes
:���������22!
sequential_1229/dense_3358/Tanh�
0sequential_1229/dense_3359/MatMul/ReadVariableOpReadVariableOp9sequential_1229_dense_3359_matmul_readvariableop_resource*
_output_shapes

:2*
dtype022
0sequential_1229/dense_3359/MatMul/ReadVariableOp�
!sequential_1229/dense_3359/MatMulMatMul#sequential_1229/dense_3358/Tanh:y:08sequential_1229/dense_3359/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!sequential_1229/dense_3359/MatMul�
1sequential_1229/dense_3359/BiasAdd/ReadVariableOpReadVariableOp:sequential_1229_dense_3359_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1229/dense_3359/BiasAdd/ReadVariableOp�
"sequential_1229/dense_3359/BiasAddBiasAdd+sequential_1229/dense_3359/MatMul:product:09sequential_1229/dense_3359/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"sequential_1229/dense_3359/BiasAdd�
IdentityIdentity+sequential_1229/dense_3359/BiasAdd:output:02^sequential_1229/dense_3358/BiasAdd/ReadVariableOp1^sequential_1229/dense_3358/MatMul/ReadVariableOp2^sequential_1229/dense_3359/BiasAdd/ReadVariableOp1^sequential_1229/dense_3359/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2f
1sequential_1229/dense_3358/BiasAdd/ReadVariableOp1sequential_1229/dense_3358/BiasAdd/ReadVariableOp2d
0sequential_1229/dense_3358/MatMul/ReadVariableOp0sequential_1229/dense_3358/MatMul/ReadVariableOp2f
1sequential_1229/dense_3359/BiasAdd/ReadVariableOp1sequential_1229/dense_3359/BiasAdd/ReadVariableOp2d
0sequential_1229/dense_3359/MatMul/ReadVariableOp0sequential_1229/dense_3359/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������
$
_user_specified_name
input_1230
�!
�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824688

inputs
dense_3358_6824665
dense_3358_6824667
dense_3359_6824670
dense_3359_6824672
identity��"dense_3358/StatefulPartitionedCall�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�"dense_3359/StatefulPartitionedCall�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
"dense_3358/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3358_6824665dense_3358_6824667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3358_layer_call_and_return_conditional_losses_68245722$
"dense_3358/StatefulPartitionedCall�
"dense_3359/StatefulPartitionedCallStatefulPartitionedCall+dense_3358/StatefulPartitionedCall:output:0dense_3359_6824670dense_3359_6824672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3359_layer_call_and_return_conditional_losses_68246042$
"dense_3359/StatefulPartitionedCall�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3358_6824665*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3359_6824670*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentity+dense_3359/StatefulPartitionedCall:output:0#^dense_3358/StatefulPartitionedCall4^dense_3358/kernel/Regularizer/Square/ReadVariableOp#^dense_3359/StatefulPartitionedCall4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2H
"dense_3358/StatefulPartitionedCall"dense_3358/StatefulPartitionedCall2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp2H
"dense_3359/StatefulPartitionedCall"dense_3359/StatefulPartitionedCall2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824727

inputs
dense_3358_6824704
dense_3358_6824706
dense_3359_6824709
dense_3359_6824711
identity��"dense_3358/StatefulPartitionedCall�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�"dense_3359/StatefulPartitionedCall�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
"dense_3358/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3358_6824704dense_3358_6824706*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3358_layer_call_and_return_conditional_losses_68245722$
"dense_3358/StatefulPartitionedCall�
"dense_3359/StatefulPartitionedCallStatefulPartitionedCall+dense_3358/StatefulPartitionedCall:output:0dense_3359_6824709dense_3359_6824711*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3359_layer_call_and_return_conditional_losses_68246042$
"dense_3359/StatefulPartitionedCall�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3358_6824704*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3359_6824709*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentity+dense_3359/StatefulPartitionedCall:output:0#^dense_3358/StatefulPartitionedCall4^dense_3358/kernel/Regularizer/Square/ReadVariableOp#^dense_3359/StatefulPartitionedCall4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2H
"dense_3358/StatefulPartitionedCall"dense_3358/StatefulPartitionedCall2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp2H
"dense_3359/StatefulPartitionedCall"dense_3359/StatefulPartitionedCall2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Z
�

#__inference__traced_restore_6825101
file_prefix&
"assignvariableop_dense_3358_kernel&
"assignvariableop_1_dense_3358_bias(
$assignvariableop_2_dense_3359_kernel&
"assignvariableop_3_dense_3359_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_10
,assignvariableop_13_adam_dense_3358_kernel_m.
*assignvariableop_14_adam_dense_3358_bias_m0
,assignvariableop_15_adam_dense_3359_kernel_m.
*assignvariableop_16_adam_dense_3359_bias_m0
,assignvariableop_17_adam_dense_3358_kernel_v.
*assignvariableop_18_adam_dense_3358_bias_v0
,assignvariableop_19_adam_dense_3359_kernel_v.
*assignvariableop_20_adam_dense_3359_bias_v
identity_22��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp"assignvariableop_dense_3358_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3358_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3359_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3359_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_dense_3358_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_3358_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_3359_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_3359_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_3358_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_3358_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_3359_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_3359_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21�
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�)
�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824831

inputs-
)dense_3358_matmul_readvariableop_resource.
*dense_3358_biasadd_readvariableop_resource-
)dense_3359_matmul_readvariableop_resource.
*dense_3359_biasadd_readvariableop_resource
identity��!dense_3358/BiasAdd/ReadVariableOp� dense_3358/MatMul/ReadVariableOp�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�!dense_3359/BiasAdd/ReadVariableOp� dense_3359/MatMul/ReadVariableOp�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
 dense_3358/MatMul/ReadVariableOpReadVariableOp)dense_3358_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02"
 dense_3358/MatMul/ReadVariableOp�
dense_3358/MatMulMatMulinputs(dense_3358/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
dense_3358/MatMul�
!dense_3358/BiasAdd/ReadVariableOpReadVariableOp*dense_3358_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02#
!dense_3358/BiasAdd/ReadVariableOp�
dense_3358/BiasAddBiasAdddense_3358/MatMul:product:0)dense_3358/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
dense_3358/BiasAddy
dense_3358/TanhTanhdense_3358/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
dense_3358/Tanh�
 dense_3359/MatMul/ReadVariableOpReadVariableOp)dense_3359_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02"
 dense_3359/MatMul/ReadVariableOp�
dense_3359/MatMulMatMuldense_3358/Tanh:y:0(dense_3359/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3359/MatMul�
!dense_3359/BiasAdd/ReadVariableOpReadVariableOp*dense_3359_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3359/BiasAdd/ReadVariableOp�
dense_3359/BiasAddBiasAdddense_3359/MatMul:product:0)dense_3359/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3359/BiasAdd�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dense_3358_matmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dense_3359_matmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentitydense_3359/BiasAdd:output:0"^dense_3358/BiasAdd/ReadVariableOp!^dense_3358/MatMul/ReadVariableOp4^dense_3358/kernel/Regularizer/Square/ReadVariableOp"^dense_3359/BiasAdd/ReadVariableOp!^dense_3359/MatMul/ReadVariableOp4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2F
!dense_3358/BiasAdd/ReadVariableOp!dense_3358/BiasAdd/ReadVariableOp2D
 dense_3358/MatMul/ReadVariableOp dense_3358/MatMul/ReadVariableOp2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp2F
!dense_3359/BiasAdd/ReadVariableOp!dense_3359/BiasAdd/ReadVariableOp2D
 dense_3359/MatMul/ReadVariableOp dense_3359/MatMul/ReadVariableOp2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824802

inputs-
)dense_3358_matmul_readvariableop_resource.
*dense_3358_biasadd_readvariableop_resource-
)dense_3359_matmul_readvariableop_resource.
*dense_3359_biasadd_readvariableop_resource
identity��!dense_3358/BiasAdd/ReadVariableOp� dense_3358/MatMul/ReadVariableOp�3dense_3358/kernel/Regularizer/Square/ReadVariableOp�!dense_3359/BiasAdd/ReadVariableOp� dense_3359/MatMul/ReadVariableOp�3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
 dense_3358/MatMul/ReadVariableOpReadVariableOp)dense_3358_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02"
 dense_3358/MatMul/ReadVariableOp�
dense_3358/MatMulMatMulinputs(dense_3358/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
dense_3358/MatMul�
!dense_3358/BiasAdd/ReadVariableOpReadVariableOp*dense_3358_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02#
!dense_3358/BiasAdd/ReadVariableOp�
dense_3358/BiasAddBiasAdddense_3358/MatMul:product:0)dense_3358/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
dense_3358/BiasAddy
dense_3358/TanhTanhdense_3358/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
dense_3358/Tanh�
 dense_3359/MatMul/ReadVariableOpReadVariableOp)dense_3359_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02"
 dense_3359/MatMul/ReadVariableOp�
dense_3359/MatMulMatMuldense_3358/Tanh:y:0(dense_3359/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3359/MatMul�
!dense_3359/BiasAdd/ReadVariableOpReadVariableOp*dense_3359_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3359/BiasAdd/ReadVariableOp�
dense_3359/BiasAddBiasAdddense_3359/MatMul:product:0)dense_3359/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3359/BiasAdd�
3dense_3358/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dense_3358_matmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3358/kernel/Regularizer/Square/ReadVariableOp�
$dense_3358/kernel/Regularizer/SquareSquare;dense_3358/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3358/kernel/Regularizer/Square�
#dense_3358/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3358/kernel/Regularizer/Const�
!dense_3358/kernel/Regularizer/SumSum(dense_3358/kernel/Regularizer/Square:y:0,dense_3358/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/Sum�
#dense_3358/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3358/kernel/Regularizer/mul/x�
!dense_3358/kernel/Regularizer/mulMul,dense_3358/kernel/Regularizer/mul/x:output:0*dense_3358/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3358/kernel/Regularizer/mul�
3dense_3359/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dense_3359_matmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3dense_3359/kernel/Regularizer/Square/ReadVariableOp�
$dense_3359/kernel/Regularizer/SquareSquare;dense_3359/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:22&
$dense_3359/kernel/Regularizer/Square�
#dense_3359/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3359/kernel/Regularizer/Const�
!dense_3359/kernel/Regularizer/SumSum(dense_3359/kernel/Regularizer/Square:y:0,dense_3359/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/Sum�
#dense_3359/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82%
#dense_3359/kernel/Regularizer/mul/x�
!dense_3359/kernel/Regularizer/mulMul,dense_3359/kernel/Regularizer/mul/x:output:0*dense_3359/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3359/kernel/Regularizer/mul�
IdentityIdentitydense_3359/BiasAdd:output:0"^dense_3358/BiasAdd/ReadVariableOp!^dense_3358/MatMul/ReadVariableOp4^dense_3358/kernel/Regularizer/Square/ReadVariableOp"^dense_3359/BiasAdd/ReadVariableOp!^dense_3359/MatMul/ReadVariableOp4^dense_3359/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2F
!dense_3358/BiasAdd/ReadVariableOp!dense_3358/BiasAdd/ReadVariableOp2D
 dense_3358/MatMul/ReadVariableOp dense_3358/MatMul/ReadVariableOp2j
3dense_3358/kernel/Regularizer/Square/ReadVariableOp3dense_3358/kernel/Regularizer/Square/ReadVariableOp2F
!dense_3359/BiasAdd/ReadVariableOp!dense_3359/BiasAdd/ReadVariableOp2D
 dense_3359/MatMul/ReadVariableOp dense_3359/MatMul/ReadVariableOp2j
3dense_3359/kernel/Regularizer/Square/ReadVariableOp3dense_3359/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_3358_layer_call_fn_6824889

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *P
fKRI
G__inference_dense_3358_layer_call_and_return_conditional_losses_68245722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_sequential_1229_layer_call_fn_6824857

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *U
fPRN
L__inference_sequential_1229_layer_call_and_return_conditional_losses_68247272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

input_12303
serving_default_input_1230:0���������>

dense_33590
StatefulPartitionedCall:0���������tensorflow/serving/predict:�p
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
<_default_save_signature
=__call__
*>&call_and_return_all_conditional_losses"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_1229", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1229", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1230"}}, {"class_name": "Dense", "config": {"name": "dense_3358", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3359", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1229", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1230"}}, {"class_name": "Dense", "config": {"name": "dense_3358", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3359", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [[{"class_name": "MeanAbsoluteError", "config": {"name": "mean_absolute_error", "dtype": "float32"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.01, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3358", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3358", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3359", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3359", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
�
iter

beta_1

beta_2
	decay
learning_rate	m4
m5m6m7	v8
v9v:v;"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
�
trainable_variables
metrics
layer_metrics
	variables
regularization_losses

layers
layer_regularization_losses
non_trainable_variables
=__call__
<_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Eserving_default"
signature_map
#:!22dense_3358/kernel
:22dense_3358/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
�
trainable_variables
metrics
 layer_metrics
	variables
regularization_losses

!layers
"layer_regularization_losses
#non_trainable_variables
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
#:!22dense_3359/kernel
:2dense_3359/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
�
trainable_variables
$metrics
%layer_metrics
	variables
regularization_losses

&layers
'layer_regularization_losses
(non_trainable_variables
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
)0
*1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	+total
	,count
-	variables
.	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	/total
	0count
1
_fn_kwargs
2	variables
3	keras_api"�
_tf_keras_metric�{"class_name": "MeanAbsoluteError", "name": "mean_absolute_error", "dtype": "float32", "config": {"name": "mean_absolute_error", "dtype": "float32"}}
:  (2total
:  (2count
.
+0
,1"
trackable_list_wrapper
-
-	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
-
2	variables"
_generic_user_object
(:&22Adam/dense_3358/kernel/m
": 22Adam/dense_3358/bias/m
(:&22Adam/dense_3359/kernel/m
": 2Adam/dense_3359/bias/m
(:&22Adam/dense_3358/kernel/v
": 22Adam/dense_3358/bias/v
(:&22Adam/dense_3359/kernel/v
": 2Adam/dense_3359/bias/v
�2�
"__inference__wrapped_model_6824551�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *)�&
$�!

input_1230���������
�2�
1__inference_sequential_1229_layer_call_fn_6824738
1__inference_sequential_1229_layer_call_fn_6824699
1__inference_sequential_1229_layer_call_fn_6824857
1__inference_sequential_1229_layer_call_fn_6824844�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824831
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824802
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824633
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824659�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dense_3358_layer_call_fn_6824889�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dense_3358_layer_call_and_return_conditional_losses_6824880�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_dense_3359_layer_call_fn_6824920�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dense_3359_layer_call_and_return_conditional_losses_6824911�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_6824931�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_6824942�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
%__inference_signature_wrapper_6824773
input_1230"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_6824551t	
3�0
)�&
$�!

input_1230���������
� "7�4
2

dense_3359$�!

dense_3359����������
G__inference_dense_3358_layer_call_and_return_conditional_losses_6824880\	
/�,
%�"
 �
inputs���������
� "%�"
�
0���������2
� 
,__inference_dense_3358_layer_call_fn_6824889O	
/�,
%�"
 �
inputs���������
� "����������2�
G__inference_dense_3359_layer_call_and_return_conditional_losses_6824911\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� 
,__inference_dense_3359_layer_call_fn_6824920O/�,
%�"
 �
inputs���������2
� "����������<
__inference_loss_fn_0_6824931	�

� 
� "� <
__inference_loss_fn_1_6824942�

� 
� "� �
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824633j	
;�8
1�.
$�!

input_1230���������
p

 
� "%�"
�
0���������
� �
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824659j	
;�8
1�.
$�!

input_1230���������
p 

 
� "%�"
�
0���������
� �
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824802f	
7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
L__inference_sequential_1229_layer_call_and_return_conditional_losses_6824831f	
7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
1__inference_sequential_1229_layer_call_fn_6824699]	
;�8
1�.
$�!

input_1230���������
p

 
� "�����������
1__inference_sequential_1229_layer_call_fn_6824738]	
;�8
1�.
$�!

input_1230���������
p 

 
� "�����������
1__inference_sequential_1229_layer_call_fn_6824844Y	
7�4
-�*
 �
inputs���������
p

 
� "�����������
1__inference_sequential_1229_layer_call_fn_6824857Y	
7�4
-�*
 �
inputs���������
p 

 
� "�����������
%__inference_signature_wrapper_6824773�	
A�>
� 
7�4
2

input_1230$�!

input_1230���������"7�4
2

dense_3359$�!

dense_3359���������