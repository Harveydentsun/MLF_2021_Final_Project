??"
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28?? 
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
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
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_4/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x**
shared_namelstm_4/lstm_cell_4/kernel
?
-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/kernel*
_output_shapes

:x*
dtype0
?
#lstm_4/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*4
shared_name%#lstm_4/lstm_cell_4/recurrent_kernel
?
7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_4/recurrent_kernel*
_output_shapes

:x*
dtype0
?
lstm_4/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*(
shared_namelstm_4/lstm_cell_4/bias

+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/bias*
_output_shapes
:x*
dtype0
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
?
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m
?
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m
?
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
?
 Adam/lstm_4/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/m
?
4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/m*
_output_shapes

:x*
dtype0
?
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
?
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m*
_output_shapes

:x*
dtype0
?
Adam/lstm_4/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*/
shared_name Adam/lstm_4/lstm_cell_4/bias/m
?
2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/m*
_output_shapes
:x*
dtype0
?
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v
?
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v
?
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
?
 Adam/lstm_4/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/v
?
4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/v*
_output_shapes

:x*
dtype0
?
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
?
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v*
_output_shapes

:x*
dtype0
?
Adam/lstm_4/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*/
shared_name Adam/lstm_4/lstm_cell_4/bias/v
?
2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/v*
_output_shapes
:x*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?6
value?6B?6 B?6
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_ratem]m^m_m` ma!mb+mc,md-mevfvgvhvi vj!vk+vl,vm-vn
N
0
1
2
3
+4
,5
-6
7
8
 9
!10
?
0
1
+2
,3
-4
5
6
 7
!8
 
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
 
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
?
8
state_size

+kernel
,recurrent_kernel
-bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
 

+0
,1
-2

+0
,1
-2
 
?

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
"	variables
#trainable_variables
$regularization_losses
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
US
VARIABLE_VALUElstm_4/lstm_cell_4/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_4/lstm_cell_4/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_4/lstm_cell_4/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
2
3

M0
N1
 
 

0
1
 
 
 
 
 

+0
,1
-2

+0
,1
-2
 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
9	variables
:trainable_variables
;regularization_losses
 
 

0
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
 
4
	Ttotal
	Ucount
V	variables
W	keras_api
D
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

V	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

[	variables
??
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_5Placeholder*+
_output_shapes
:?????????	*
dtype0* 
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betalstm_4/lstm_cell_4/kernellstm_4/lstm_cell_4/bias#lstm_4/lstm_cell_4/recurrent_kerneldense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1895752
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpConst*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1898040
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biastotalcounttotal_1count_1"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m Adam/lstm_4/lstm_cell_4/kernel/m*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mAdam/lstm_4/lstm_cell_4/bias/m"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v Adam/lstm_4/lstm_cell_4/kernel/v*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vAdam/lstm_4/lstm_cell_4/bias/v*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1898164??
??
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897648

inputs;
)lstm_cell_4_split_readvariableop_resource:x9
+lstm_cell_4_split_1_readvariableop_resource:x5
#lstm_cell_4_readvariableop_resource:x
identity??lstm_cell_4/ReadVariableOp?lstm_cell_4/ReadVariableOp_1?lstm_cell_4/ReadVariableOp_2?lstm_cell_4/ReadVariableOp_3? lstm_cell_4/split/ReadVariableOp?"lstm_cell_4/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????^
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????g
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???g
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??Di
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2̰?i
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???i
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????}
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1897489*
condR
while_cond_1897488*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?l
?	
while_body_1896706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_4_split_readvariableop_resource_0:xA
3while_lstm_cell_4_split_1_readvariableop_resource_0:x=
+while_lstm_cell_4_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_4_split_readvariableop_resource:x?
1while_lstm_cell_4_split_1_readvariableop_resource:x;
)while_lstm_cell_4_readvariableop_resource:x?? while/lstm_cell_4/ReadVariableOp?"while/lstm_cell_4/ReadVariableOp_1?"while/lstm_cell_4/ReadVariableOp_2?"while/lstm_cell_4/ReadVariableOp_3?&while/lstm_cell_4/split/ReadVariableOp?(while/lstm_cell_4/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896526

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????	?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????	: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
-__inference_lstm_cell_4_layer_call_fn_1897721

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?

?
.__inference_sequential_4_layer_call_fn_1895657
input_5
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?
?
while_cond_1894741
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1894741___redundant_placeholder05
1while_while_cond_1894741___redundant_placeholder15
1while_while_cond_1894741___redundant_placeholder25
1while_while_cond_1894741___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?

I__inference_sequential_4_layer_call_and_return_conditional_losses_1896064

inputsE
7batch_normalization_4_batchnorm_readvariableop_resource:I
;batch_normalization_4_batchnorm_mul_readvariableop_resource:G
9batch_normalization_4_batchnorm_readvariableop_1_resource:G
9batch_normalization_4_batchnorm_readvariableop_2_resource:B
0lstm_4_lstm_cell_4_split_readvariableop_resource:x@
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:x<
*lstm_4_lstm_cell_4_readvariableop_resource:x8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity??.batch_normalization_4/batchnorm/ReadVariableOp?0batch_normalization_4/batchnorm/ReadVariableOp_1?0batch_normalization_4/batchnorm/ReadVariableOp_2?2batch_normalization_4/batchnorm/mul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?!lstm_4/lstm_cell_4/ReadVariableOp?#lstm_4/lstm_cell_4/ReadVariableOp_1?#lstm_4/lstm_cell_4/ReadVariableOp_2?#lstm_4/lstm_cell_4/ReadVariableOp_3?'lstm_4/lstm_cell_4/split/ReadVariableOp?)lstm_4/lstm_cell_4/split_1/ReadVariableOp?lstm_4/while?
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_4/batchnorm/mul_1Mulinputs'batch_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	?
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:?
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	e
lstm_4/ShapeShape)batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Y
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????j
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_4/transpose	Transpose)batch_normalization_4/batchnorm/add_1:z:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:	?????????R
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskg
"lstm_4/lstm_cell_4/ones_like/ShapeShapelstm_4/zeros:output:0*
T0*
_output_shapes
:g
"lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_4/lstm_cell_4/ones_likeFill+lstm_4/lstm_cell_4/ones_like/Shape:output:0+lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????d
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????f
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0w
&lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_4/lstm_cell_4/strided_sliceStridedSlice)lstm_4/lstm_cell_4/ReadVariableOp:value:0/lstm_4/lstm_cell_4/strided_slice/stack:output:01lstm_4/lstm_cell_4/strided_slice/stack_1:output:01lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????s
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   {
*lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_4/lstm_cell_4/strided_slice_1StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_1:value:01lstm_4/lstm_cell_4/strided_slice_1/stack:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????w
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:??????????
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   {
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   {
*lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_4/lstm_cell_4/strided_slice_2StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_2:value:01lstm_4/lstm_cell_4/strided_slice_2/stack:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????o
lstm_4/lstm_cell_4/TanhTanhlstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0lstm_4/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   {
*lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_4/lstm_cell_4/strided_slice_3StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_3:value:01lstm_4/lstm_cell_4/strided_slice_3/stack:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????w
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????q
lstm_4/lstm_cell_4/Tanh_1Tanhlstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0lstm_4/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????u
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_4_lstm_cell_4_split_readvariableop_resource2lstm_4_lstm_cell_4_split_1_readvariableop_resource*lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_4_while_body_1895924*%
condR
lstm_4_while_cond_1895923*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0o
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	b
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_8/MatMulMatMullstm_4/strided_slice_3:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2F
!lstm_4/lstm_cell_4/ReadVariableOp!lstm_4/lstm_cell_4/ReadVariableOp2J
#lstm_4/lstm_cell_4/ReadVariableOp_1#lstm_4/lstm_cell_4/ReadVariableOp_12J
#lstm_4/lstm_cell_4/ReadVariableOp_2#lstm_4/lstm_cell_4/ReadVariableOp_22J
#lstm_4/lstm_cell_4/ReadVariableOp_3#lstm_4/lstm_cell_4/ReadVariableOp_32R
'lstm_4/lstm_cell_4/split/ReadVariableOp'lstm_4/lstm_cell_4/split/ReadVariableOp2V
)lstm_4/lstm_cell_4/split_1/ReadVariableOp)lstm_4/lstm_cell_4/split_1/ReadVariableOp2
lstm_4/whilelstm_4/while:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
ޔ
?
&sequential_4_lstm_4_while_body_1894130D
@sequential_4_lstm_4_while_sequential_4_lstm_4_while_loop_counterJ
Fsequential_4_lstm_4_while_sequential_4_lstm_4_while_maximum_iterations)
%sequential_4_lstm_4_while_placeholder+
'sequential_4_lstm_4_while_placeholder_1+
'sequential_4_lstm_4_while_placeholder_2+
'sequential_4_lstm_4_while_placeholder_3C
?sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1_0
{sequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor_0W
Esequential_4_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:xU
Gsequential_4_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:xQ
?sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource_0:x&
"sequential_4_lstm_4_while_identity(
$sequential_4_lstm_4_while_identity_1(
$sequential_4_lstm_4_while_identity_2(
$sequential_4_lstm_4_while_identity_3(
$sequential_4_lstm_4_while_identity_4(
$sequential_4_lstm_4_while_identity_5A
=sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1}
ysequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensorU
Csequential_4_lstm_4_while_lstm_cell_4_split_readvariableop_resource:xS
Esequential_4_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:xO
=sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource:x??4sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp?6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_1?6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_2?6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_3?:sequential_4/lstm_4/while/lstm_cell_4/split/ReadVariableOp?<sequential_4/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp?
Ksequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
=sequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_4_while_placeholderTsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
5sequential_4/lstm_4/while/lstm_cell_4/ones_like/ShapeShape'sequential_4_lstm_4_while_placeholder_2*
T0*
_output_shapes
:z
5sequential_4/lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
/sequential_4/lstm_4/while/lstm_cell_4/ones_likeFill>sequential_4/lstm_4/while/lstm_cell_4/ones_like/Shape:output:0>sequential_4/lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????w
5sequential_4/lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
:sequential_4/lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOpEsequential_4_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
+sequential_4/lstm_4/while/lstm_cell_4/splitSplit>sequential_4/lstm_4/while/lstm_cell_4/split/split_dim:output:0Bsequential_4/lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
,sequential_4/lstm_4/while/lstm_cell_4/MatMulMatMulDsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_1MatMulDsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_2MatMulDsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_3MatMulDsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_4/lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????y
7sequential_4/lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_4/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOpGsequential_4_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
-sequential_4/lstm_4/while/lstm_cell_4/split_1Split@sequential_4/lstm_4/while/lstm_cell_4/split_1/split_dim:output:0Dsequential_4/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
-sequential_4/lstm_4/while/lstm_cell_4/BiasAddBiasAdd6sequential_4/lstm_4/while/lstm_cell_4/MatMul:product:06sequential_4/lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
/sequential_4/lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd8sequential_4/lstm_4/while/lstm_cell_4/MatMul_1:product:06sequential_4/lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
/sequential_4/lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd8sequential_4/lstm_4/while/lstm_cell_4/MatMul_2:product:06sequential_4/lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
/sequential_4/lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd8sequential_4/lstm_4/while/lstm_cell_4/MatMul_3:product:06sequential_4/lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/while/lstm_cell_4/mulMul'sequential_4_lstm_4_while_placeholder_28sequential_4/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/mul_1Mul'sequential_4_lstm_4_while_placeholder_28sequential_4/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/mul_2Mul'sequential_4_lstm_4_while_placeholder_28sequential_4/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/mul_3Mul'sequential_4_lstm_4_while_placeholder_28sequential_4/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
4sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp?sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
9sequential_4/lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
;sequential_4/lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
;sequential_4/lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
3sequential_4/lstm_4/while/lstm_cell_4/strided_sliceStridedSlice<sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp:value:0Bsequential_4/lstm_4/while/lstm_cell_4/strided_slice/stack:output:0Dsequential_4/lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:0Dsequential_4/lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_4MatMul-sequential_4/lstm_4/while/lstm_cell_4/mul:z:0<sequential_4/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/while/lstm_cell_4/addAddV26sequential_4/lstm_4/while/lstm_cell_4/BiasAdd:output:08sequential_4/lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:??????????
-sequential_4/lstm_4/while/lstm_cell_4/SigmoidSigmoid-sequential_4/lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp?sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
;sequential_4/lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_4/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   ?
=sequential_4/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_4/lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice>sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:0Dsequential_4/lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:0Fsequential_4/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:0Fsequential_4/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_5MatMul/sequential_4/lstm_4/while/lstm_cell_4/mul_1:z:0>sequential_4/lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/add_1AddV28sequential_4/lstm_4/while/lstm_cell_4/BiasAdd_1:output:08sequential_4/lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
/sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid/sequential_4/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/mul_4Mul3sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_1:y:0'sequential_4_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:??????????
6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp?sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
;sequential_4/lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   ?
=sequential_4/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   ?
=sequential_4/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_4/lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice>sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:0Dsequential_4/lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:0Fsequential_4/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:0Fsequential_4/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_6MatMul/sequential_4/lstm_4/while/lstm_cell_4/mul_2:z:0>sequential_4/lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/add_2AddV28sequential_4/lstm_4/while/lstm_cell_4/BiasAdd_2:output:08sequential_4/lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:??????????
*sequential_4/lstm_4/while/lstm_cell_4/TanhTanh/sequential_4/lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/mul_5Mul1sequential_4/lstm_4/while/lstm_cell_4/Sigmoid:y:0.sequential_4/lstm_4/while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/add_3AddV2/sequential_4/lstm_4/while/lstm_cell_4/mul_4:z:0/sequential_4/lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp?sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
;sequential_4/lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   ?
=sequential_4/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_4/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_4/lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice>sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:0Dsequential_4/lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:0Fsequential_4/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:0Fsequential_4/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_7MatMul/sequential_4/lstm_4/while/lstm_cell_4/mul_3:z:0>sequential_4/lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/add_4AddV28sequential_4/lstm_4/while/lstm_cell_4/BiasAdd_3:output:08sequential_4/lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
/sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid/sequential_4/lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:??????????
,sequential_4/lstm_4/while/lstm_cell_4/Tanh_1Tanh/sequential_4/lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
+sequential_4/lstm_4/while/lstm_cell_4/mul_6Mul3sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_2:y:00sequential_4/lstm_4/while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
>sequential_4/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_4_while_placeholder_1%sequential_4_lstm_4_while_placeholder/sequential_4/lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???a
sequential_4/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_4/lstm_4/while/addAddV2%sequential_4_lstm_4_while_placeholder(sequential_4/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_4/lstm_4/while/add_1AddV2@sequential_4_lstm_4_while_sequential_4_lstm_4_while_loop_counter*sequential_4/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: ?
"sequential_4/lstm_4/while/IdentityIdentity#sequential_4/lstm_4/while/add_1:z:0^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: ?
$sequential_4/lstm_4/while/Identity_1IdentityFsequential_4_lstm_4_while_sequential_4_lstm_4_while_maximum_iterations^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: ?
$sequential_4/lstm_4/while/Identity_2Identity!sequential_4/lstm_4/while/add:z:0^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: ?
$sequential_4/lstm_4/while/Identity_3IdentityNsequential_4/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: ?
$sequential_4/lstm_4/while/Identity_4Identity/sequential_4/lstm_4/while/lstm_cell_4/mul_6:z:0^sequential_4/lstm_4/while/NoOp*
T0*'
_output_shapes
:??????????
$sequential_4/lstm_4/while/Identity_5Identity/sequential_4/lstm_4/while/lstm_cell_4/add_3:z:0^sequential_4/lstm_4/while/NoOp*
T0*'
_output_shapes
:??????????
sequential_4/lstm_4/while/NoOpNoOp5^sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp7^sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_17^sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_27^sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_3;^sequential_4/lstm_4/while/lstm_cell_4/split/ReadVariableOp=^sequential_4/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_4_lstm_4_while_identity+sequential_4/lstm_4/while/Identity:output:0"U
$sequential_4_lstm_4_while_identity_1-sequential_4/lstm_4/while/Identity_1:output:0"U
$sequential_4_lstm_4_while_identity_2-sequential_4/lstm_4/while/Identity_2:output:0"U
$sequential_4_lstm_4_while_identity_3-sequential_4/lstm_4/while/Identity_3:output:0"U
$sequential_4_lstm_4_while_identity_4-sequential_4/lstm_4/while/Identity_4:output:0"U
$sequential_4_lstm_4_while_identity_5-sequential_4/lstm_4/while/Identity_5:output:0"?
=sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource?sequential_4_lstm_4_while_lstm_cell_4_readvariableop_resource_0"?
Esequential_4_lstm_4_while_lstm_cell_4_split_1_readvariableop_resourceGsequential_4_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"?
Csequential_4_lstm_4_while_lstm_cell_4_split_readvariableop_resourceEsequential_4_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"?
=sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1?sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1_0"?
ysequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2l
4sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp4sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp2p
6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_16sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_12p
6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_26sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_22p
6sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_36sequential_4/lstm_4/while/lstm_cell_4/ReadVariableOp_32x
:sequential_4/lstm_4/while/lstm_cell_4/split/ReadVariableOp:sequential_4/lstm_4/while/lstm_cell_4/split/ReadVariableOp2|
<sequential_4/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp<sequential_4/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
7__inference_batch_normalization_4_layer_call_fn_1896426

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894341|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?>
?
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1897796

inputs
states_0
states_1/
split_readvariableop_resource:x-
split_1_readvariableop_resource:x)
readvariableop_resource:x
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:x*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????Z
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????\
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????\
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????\
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Y
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?%
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894341

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895687
input_5+
batch_normalization_4_1895660:+
batch_normalization_4_1895662:+
batch_normalization_4_1895664:+
batch_normalization_4_1895666: 
lstm_4_1895669:x
lstm_4_1895671:x 
lstm_4_1895673:x!
dense_8_1895676:
dense_8_1895678:!
dense_9_1895681:
dense_9_1895683:
identity??-batch_normalization_4/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinput_5batch_normalization_4_1895660batch_normalization_4_1895662batch_normalization_4_1895664batch_normalization_4_1895666*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894846?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0lstm_4_1895669lstm_4_1895671lstm_4_1895673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895084?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_1895676dense_8_1895678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1895103?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1895681dense_9_1895683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1895119w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?
?
while_cond_1896705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1896705___redundant_placeholder05
1while_while_cond_1896705___redundant_placeholder15
1while_while_cond_1896705___redundant_placeholder25
1while_while_cond_1896705___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1894475
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1894475___redundant_placeholder05
1while_while_cond_1894475___redundant_placeholder15
1while_while_cond_1894475___redundant_placeholder25
1while_while_cond_1894475___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894846

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????	?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????	: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?y
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895084

inputs;
)lstm_cell_4_split_readvariableop_resource:x9
+lstm_cell_4_split_1_readvariableop_resource:x5
#lstm_cell_4_readvariableop_resource:x
identity??lstm_cell_4/ReadVariableOp?lstm_cell_4/ReadVariableOp_1?lstm_cell_4/ReadVariableOp_2?lstm_cell_4/ReadVariableOp_3? lstm_cell_4/split/ReadVariableOp?"lstm_cell_4/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:?????????x
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????~
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????}
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1894957*
condR
while_cond_1894956*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
while_cond_1897488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1897488___redundant_placeholder05
1while_while_cond_1897488___redundant_placeholder15
1while_while_cond_1897488___redundant_placeholder25
1while_while_cond_1897488___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?

?
.__inference_sequential_4_layer_call_fn_1895806

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
while_cond_1897227
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1897227___redundant_placeholder05
1while_while_cond_1897227___redundant_placeholder15
1while_while_cond_1897227___redundant_placeholder25
1while_while_cond_1897227___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?

?
%__inference_signature_wrapper_1895752
input_5
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_1894270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894294

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?\
?
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1897903

inputs
states_0
states_1/
split_readvariableop_resource:x-
split_1_readvariableop_resource:x)
readvariableop_resource:x
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??Q]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2?Ê]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:x*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????Y
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????]
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????]
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????]
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Y
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
while_cond_1894956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1894956___redundant_placeholder05
1while_while_cond_1894956___redundant_placeholder15
1while_while_cond_1894956___redundant_placeholder25
1while_while_cond_1894956___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
lstm_4_while_cond_1895923*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1C
?lstm_4_while_lstm_4_while_cond_1895923___redundant_placeholder0C
?lstm_4_while_lstm_4_while_cond_1895923___redundant_placeholder1C
?lstm_4_while_lstm_4_while_cond_1895923___redundant_placeholder2C
?lstm_4_while_lstm_4_while_cond_1895923___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_dense_9_layer_call_fn_1897677

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1895119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ٓ
?	
while_body_1895318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_4_split_readvariableop_resource_0:xA
3while_lstm_cell_4_split_1_readvariableop_resource_0:x=
+while_lstm_cell_4_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_4_split_readvariableop_resource:x?
1while_lstm_cell_4_split_1_readvariableop_resource:x;
)while_lstm_cell_4_readvariableop_resource:x?? while/lstm_cell_4/ReadVariableOp?"while/lstm_cell_4/ReadVariableOp_1?"while/lstm_cell_4/ReadVariableOp_2?"while/lstm_cell_4/ReadVariableOp_3?&while/lstm_cell_4/split/ReadVariableOp?(while/lstm_cell_4/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????d
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????s
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2?Ҋm
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???o
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2õ?o
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???o
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ߙ
?
#__inference__traced_restore_1898164
file_prefix:
,assignvariableop_batch_normalization_4_gamma:;
-assignvariableop_1_batch_normalization_4_beta:B
4assignvariableop_2_batch_normalization_4_moving_mean:F
8assignvariableop_3_batch_normalization_4_moving_variance:3
!assignvariableop_4_dense_8_kernel:-
assignvariableop_5_dense_8_bias:3
!assignvariableop_6_dense_9_kernel:-
assignvariableop_7_dense_9_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: ?
-assignvariableop_13_lstm_4_lstm_cell_4_kernel:xI
7assignvariableop_14_lstm_4_lstm_cell_4_recurrent_kernel:x9
+assignvariableop_15_lstm_4_lstm_cell_4_bias:x#
assignvariableop_16_total: #
assignvariableop_17_count: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: D
6assignvariableop_20_adam_batch_normalization_4_gamma_m:C
5assignvariableop_21_adam_batch_normalization_4_beta_m:;
)assignvariableop_22_adam_dense_8_kernel_m:5
'assignvariableop_23_adam_dense_8_bias_m:;
)assignvariableop_24_adam_dense_9_kernel_m:5
'assignvariableop_25_adam_dense_9_bias_m:F
4assignvariableop_26_adam_lstm_4_lstm_cell_4_kernel_m:xP
>assignvariableop_27_adam_lstm_4_lstm_cell_4_recurrent_kernel_m:x@
2assignvariableop_28_adam_lstm_4_lstm_cell_4_bias_m:xD
6assignvariableop_29_adam_batch_normalization_4_gamma_v:C
5assignvariableop_30_adam_batch_normalization_4_beta_v:;
)assignvariableop_31_adam_dense_8_kernel_v:5
'assignvariableop_32_adam_dense_8_bias_v:;
)assignvariableop_33_adam_dense_9_kernel_v:5
'assignvariableop_34_adam_dense_9_bias_v:F
4assignvariableop_35_adam_lstm_4_lstm_cell_4_kernel_v:xP
>assignvariableop_36_adam_lstm_4_lstm_cell_4_recurrent_kernel_v:x@
2assignvariableop_37_adam_lstm_4_lstm_cell_4_bias_v:x
identity_39??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_4_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_4_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_4_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_4_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_8_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_8_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_4_lstm_cell_4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_lstm_4_lstm_cell_4_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_lstm_4_lstm_cell_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_batch_normalization_4_gamma_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_batch_normalization_4_beta_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_8_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_8_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_9_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_9_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_4_lstm_cell_4_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_lstm_4_lstm_cell_4_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_lstm_4_lstm_cell_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_batch_normalization_4_gamma_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_batch_normalization_4_beta_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_8_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_8_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_9_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_9_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_4_lstm_cell_4_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp>assignvariableop_36_adam_lstm_4_lstm_cell_4_recurrent_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_lstm_4_lstm_cell_4_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_39IdentityIdentity_38:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_39Identity_39:output:0*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
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
?	
?
D__inference_dense_9_layer_call_and_return_conditional_losses_1895119

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_1894270
input_5R
Dsequential_4_batch_normalization_4_batchnorm_readvariableop_resource:V
Hsequential_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:T
Fsequential_4_batch_normalization_4_batchnorm_readvariableop_1_resource:T
Fsequential_4_batch_normalization_4_batchnorm_readvariableop_2_resource:O
=sequential_4_lstm_4_lstm_cell_4_split_readvariableop_resource:xM
?sequential_4_lstm_4_lstm_cell_4_split_1_readvariableop_resource:xI
7sequential_4_lstm_4_lstm_cell_4_readvariableop_resource:xE
3sequential_4_dense_8_matmul_readvariableop_resource:B
4sequential_4_dense_8_biasadd_readvariableop_resource:E
3sequential_4_dense_9_matmul_readvariableop_resource:B
4sequential_4_dense_9_biasadd_readvariableop_resource:
identity??;sequential_4/batch_normalization_4/batchnorm/ReadVariableOp?=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_1?=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_2??sequential_4/batch_normalization_4/batchnorm/mul/ReadVariableOp?+sequential_4/dense_8/BiasAdd/ReadVariableOp?*sequential_4/dense_8/MatMul/ReadVariableOp?+sequential_4/dense_9/BiasAdd/ReadVariableOp?*sequential_4/dense_9/MatMul/ReadVariableOp?.sequential_4/lstm_4/lstm_cell_4/ReadVariableOp?0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_1?0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_2?0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_3?4sequential_4/lstm_4/lstm_cell_4/split/ReadVariableOp?6sequential_4/lstm_4/lstm_cell_4/split_1/ReadVariableOp?sequential_4/lstm_4/while?
;sequential_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpDsequential_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0w
2sequential_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
0sequential_4/batch_normalization_4/batchnorm/addAddV2Csequential_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0;sequential_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
2sequential_4/batch_normalization_4/batchnorm/RsqrtRsqrt4sequential_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:?
?sequential_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
0sequential_4/batch_normalization_4/batchnorm/mulMul6sequential_4/batch_normalization_4/batchnorm/Rsqrt:y:0Gsequential_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
2sequential_4/batch_normalization_4/batchnorm/mul_1Mulinput_54sequential_4/batch_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	?
=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_4_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
2sequential_4/batch_normalization_4/batchnorm/mul_2MulEsequential_4/batch_normalization_4/batchnorm/ReadVariableOp_1:value:04sequential_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:?
=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_4_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
0sequential_4/batch_normalization_4/batchnorm/subSubEsequential_4/batch_normalization_4/batchnorm/ReadVariableOp_2:value:06sequential_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
2sequential_4/batch_normalization_4/batchnorm/add_1AddV26sequential_4/batch_normalization_4/batchnorm/mul_1:z:04sequential_4/batch_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	
sequential_4/lstm_4/ShapeShape6sequential_4/batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:q
'sequential_4/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential_4/lstm_4/strided_sliceStridedSlice"sequential_4/lstm_4/Shape:output:00sequential_4/lstm_4/strided_slice/stack:output:02sequential_4/lstm_4/strided_slice/stack_1:output:02sequential_4/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_4/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
 sequential_4/lstm_4/zeros/packedPack*sequential_4/lstm_4/strided_slice:output:0+sequential_4/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_4/lstm_4/zerosFill)sequential_4/lstm_4/zeros/packed:output:0(sequential_4/lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????f
$sequential_4/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
"sequential_4/lstm_4/zeros_1/packedPack*sequential_4/lstm_4/strided_slice:output:0-sequential_4/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_4/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_4/lstm_4/zeros_1Fill+sequential_4/lstm_4/zeros_1/packed:output:0*sequential_4/lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????w
"sequential_4/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_4/lstm_4/transpose	Transpose6sequential_4/batch_normalization_4/batchnorm/add_1:z:0+sequential_4/lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:	?????????l
sequential_4/lstm_4/Shape_1Shape!sequential_4/lstm_4/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_4/lstm_4/strided_slice_1StridedSlice$sequential_4/lstm_4/Shape_1:output:02sequential_4/lstm_4/strided_slice_1/stack:output:04sequential_4/lstm_4/strided_slice_1/stack_1:output:04sequential_4/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!sequential_4/lstm_4/TensorArrayV2TensorListReserve8sequential_4/lstm_4/TensorArrayV2/element_shape:output:0,sequential_4/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Isequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
;sequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_4/transpose:y:0Rsequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???s
)sequential_4/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_4/lstm_4/strided_slice_2StridedSlice!sequential_4/lstm_4/transpose:y:02sequential_4/lstm_4/strided_slice_2/stack:output:04sequential_4/lstm_4/strided_slice_2/stack_1:output:04sequential_4/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
/sequential_4/lstm_4/lstm_cell_4/ones_like/ShapeShape"sequential_4/lstm_4/zeros:output:0*
T0*
_output_shapes
:t
/sequential_4/lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)sequential_4/lstm_4/lstm_cell_4/ones_likeFill8sequential_4/lstm_4/lstm_cell_4/ones_like/Shape:output:08sequential_4/lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????q
/sequential_4/lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_4/lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp=sequential_4_lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
%sequential_4/lstm_4/lstm_cell_4/splitSplit8sequential_4/lstm_4/lstm_cell_4/split/split_dim:output:0<sequential_4/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
&sequential_4/lstm_4/lstm_cell_4/MatMulMatMul,sequential_4/lstm_4/strided_slice_2:output:0.sequential_4/lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
(sequential_4/lstm_4/lstm_cell_4/MatMul_1MatMul,sequential_4/lstm_4/strided_slice_2:output:0.sequential_4/lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
(sequential_4/lstm_4/lstm_cell_4/MatMul_2MatMul,sequential_4/lstm_4/strided_slice_2:output:0.sequential_4/lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
(sequential_4/lstm_4/lstm_cell_4/MatMul_3MatMul,sequential_4/lstm_4/strided_slice_2:output:0.sequential_4/lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????s
1sequential_4/lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_4/lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp?sequential_4_lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
'sequential_4/lstm_4/lstm_cell_4/split_1Split:sequential_4/lstm_4/lstm_cell_4/split_1/split_dim:output:0>sequential_4/lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
'sequential_4/lstm_4/lstm_cell_4/BiasAddBiasAdd0sequential_4/lstm_4/lstm_cell_4/MatMul:product:00sequential_4/lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/lstm_cell_4/BiasAdd_1BiasAdd2sequential_4/lstm_4/lstm_cell_4/MatMul_1:product:00sequential_4/lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/lstm_cell_4/BiasAdd_2BiasAdd2sequential_4/lstm_4/lstm_cell_4/MatMul_2:product:00sequential_4/lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/lstm_cell_4/BiasAdd_3BiasAdd2sequential_4/lstm_4/lstm_cell_4/MatMul_3:product:00sequential_4/lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
#sequential_4/lstm_4/lstm_cell_4/mulMul"sequential_4/lstm_4/zeros:output:02sequential_4/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/mul_1Mul"sequential_4/lstm_4/zeros:output:02sequential_4/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/mul_2Mul"sequential_4/lstm_4/zeros:output:02sequential_4/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/mul_3Mul"sequential_4/lstm_4/zeros:output:02sequential_4/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
.sequential_4/lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp7sequential_4_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0?
3sequential_4/lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
5sequential_4/lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
5sequential_4/lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
-sequential_4/lstm_4/lstm_cell_4/strided_sliceStridedSlice6sequential_4/lstm_4/lstm_cell_4/ReadVariableOp:value:0<sequential_4/lstm_4/lstm_cell_4/strided_slice/stack:output:0>sequential_4/lstm_4/lstm_cell_4/strided_slice/stack_1:output:0>sequential_4/lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_4/lstm_4/lstm_cell_4/MatMul_4MatMul'sequential_4/lstm_4/lstm_cell_4/mul:z:06sequential_4/lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
#sequential_4/lstm_4/lstm_cell_4/addAddV20sequential_4/lstm_4/lstm_cell_4/BiasAdd:output:02sequential_4/lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:??????????
'sequential_4/lstm_4/lstm_cell_4/SigmoidSigmoid'sequential_4/lstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp7sequential_4_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0?
5sequential_4/lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_4/lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   ?
7sequential_4/lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_4/lstm_4/lstm_cell_4/strided_slice_1StridedSlice8sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_1:value:0>sequential_4/lstm_4/lstm_cell_4/strided_slice_1/stack:output:0@sequential_4/lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:0@sequential_4/lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_4/lstm_4/lstm_cell_4/MatMul_5MatMul)sequential_4/lstm_4/lstm_cell_4/mul_1:z:08sequential_4/lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/add_1AddV22sequential_4/lstm_4/lstm_cell_4/BiasAdd_1:output:02sequential_4/lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/lstm_cell_4/Sigmoid_1Sigmoid)sequential_4/lstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/mul_4Mul-sequential_4/lstm_4/lstm_cell_4/Sigmoid_1:y:0$sequential_4/lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:??????????
0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp7sequential_4_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0?
5sequential_4/lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   ?
7sequential_4/lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   ?
7sequential_4/lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_4/lstm_4/lstm_cell_4/strided_slice_2StridedSlice8sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_2:value:0>sequential_4/lstm_4/lstm_cell_4/strided_slice_2/stack:output:0@sequential_4/lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:0@sequential_4/lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_4/lstm_4/lstm_cell_4/MatMul_6MatMul)sequential_4/lstm_4/lstm_cell_4/mul_2:z:08sequential_4/lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/add_2AddV22sequential_4/lstm_4/lstm_cell_4/BiasAdd_2:output:02sequential_4/lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:??????????
$sequential_4/lstm_4/lstm_cell_4/TanhTanh)sequential_4/lstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/mul_5Mul+sequential_4/lstm_4/lstm_cell_4/Sigmoid:y:0(sequential_4/lstm_4/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/add_3AddV2)sequential_4/lstm_4/lstm_cell_4/mul_4:z:0)sequential_4/lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp7sequential_4_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0?
5sequential_4/lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   ?
7sequential_4/lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_4/lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_4/lstm_4/lstm_cell_4/strided_slice_3StridedSlice8sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_3:value:0>sequential_4/lstm_4/lstm_cell_4/strided_slice_3/stack:output:0@sequential_4/lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:0@sequential_4/lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_4/lstm_4/lstm_cell_4/MatMul_7MatMul)sequential_4/lstm_4/lstm_cell_4/mul_3:z:08sequential_4/lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/add_4AddV22sequential_4/lstm_4/lstm_cell_4/BiasAdd_3:output:02sequential_4/lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
)sequential_4/lstm_4/lstm_cell_4/Sigmoid_2Sigmoid)sequential_4/lstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:??????????
&sequential_4/lstm_4/lstm_cell_4/Tanh_1Tanh)sequential_4/lstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
%sequential_4/lstm_4/lstm_cell_4/mul_6Mul-sequential_4/lstm_4/lstm_cell_4/Sigmoid_2:y:0*sequential_4/lstm_4/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
1sequential_4/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#sequential_4/lstm_4/TensorArrayV2_1TensorListReserve:sequential_4/lstm_4/TensorArrayV2_1/element_shape:output:0,sequential_4/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???Z
sequential_4/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????h
&sequential_4/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_4/lstm_4/whileWhile/sequential_4/lstm_4/while/loop_counter:output:05sequential_4/lstm_4/while/maximum_iterations:output:0!sequential_4/lstm_4/time:output:0,sequential_4/lstm_4/TensorArrayV2_1:handle:0"sequential_4/lstm_4/zeros:output:0$sequential_4/lstm_4/zeros_1:output:0,sequential_4/lstm_4/strided_slice_1:output:0Ksequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_4_lstm_4_lstm_cell_4_split_readvariableop_resource?sequential_4_lstm_4_lstm_cell_4_split_1_readvariableop_resource7sequential_4_lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_4_lstm_4_while_body_1894130*2
cond*R(
&sequential_4_lstm_4_while_cond_1894129*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
Dsequential_4/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6sequential_4/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_4/while:output:3Msequential_4/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0|
)sequential_4/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????u
+sequential_4/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_4/lstm_4/strided_slice_3StridedSlice?sequential_4/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_4/strided_slice_3/stack:output:04sequential_4/lstm_4/strided_slice_3/stack_1:output:04sequential_4/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_masky
$sequential_4/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_4/lstm_4/transpose_1	Transpose?sequential_4/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	o
sequential_4/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_4/dense_8/MatMulMatMul,sequential_4/lstm_4/strided_slice_3:output:02sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_4/dense_8/BiasAddBiasAdd%sequential_4/dense_8/MatMul:product:03sequential_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
sequential_4/dense_8/ReluRelu%sequential_4/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
*sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_4/dense_9/MatMulMatMul'sequential_4/dense_8/Relu:activations:02sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_4/dense_9/BiasAddBiasAdd%sequential_4/dense_9/MatMul:product:03sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_4/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp<^sequential_4/batch_normalization_4/batchnorm/ReadVariableOp>^sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_1>^sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_2@^sequential_4/batch_normalization_4/batchnorm/mul/ReadVariableOp,^sequential_4/dense_8/BiasAdd/ReadVariableOp+^sequential_4/dense_8/MatMul/ReadVariableOp,^sequential_4/dense_9/BiasAdd/ReadVariableOp+^sequential_4/dense_9/MatMul/ReadVariableOp/^sequential_4/lstm_4/lstm_cell_4/ReadVariableOp1^sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_11^sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_21^sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_35^sequential_4/lstm_4/lstm_cell_4/split/ReadVariableOp7^sequential_4/lstm_4/lstm_cell_4/split_1/ReadVariableOp^sequential_4/lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2z
;sequential_4/batch_normalization_4/batchnorm/ReadVariableOp;sequential_4/batch_normalization_4/batchnorm/ReadVariableOp2~
=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_1=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_12~
=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_2=sequential_4/batch_normalization_4/batchnorm/ReadVariableOp_22?
?sequential_4/batch_normalization_4/batchnorm/mul/ReadVariableOp?sequential_4/batch_normalization_4/batchnorm/mul/ReadVariableOp2Z
+sequential_4/dense_8/BiasAdd/ReadVariableOp+sequential_4/dense_8/BiasAdd/ReadVariableOp2X
*sequential_4/dense_8/MatMul/ReadVariableOp*sequential_4/dense_8/MatMul/ReadVariableOp2Z
+sequential_4/dense_9/BiasAdd/ReadVariableOp+sequential_4/dense_9/BiasAdd/ReadVariableOp2X
*sequential_4/dense_9/MatMul/ReadVariableOp*sequential_4/dense_9/MatMul/ReadVariableOp2`
.sequential_4/lstm_4/lstm_cell_4/ReadVariableOp.sequential_4/lstm_4/lstm_cell_4/ReadVariableOp2d
0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_10sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_12d
0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_20sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_22d
0sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_30sequential_4/lstm_4/lstm_cell_4/ReadVariableOp_32l
4sequential_4/lstm_4/lstm_cell_4/split/ReadVariableOp4sequential_4/lstm_4/lstm_cell_4/split/ReadVariableOp2p
6sequential_4/lstm_4/lstm_cell_4/split_1/ReadVariableOp6sequential_4/lstm_4/lstm_cell_4/split_1/ReadVariableOp26
sequential_4/lstm_4/whilesequential_4/lstm_4/while:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?"
?
while_body_1894742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_1894766_0:x)
while_lstm_cell_4_1894768_0:x-
while_lstm_cell_4_1894770_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_1894766:x'
while_lstm_cell_4_1894768:x+
while_lstm_cell_4_1894770:x??)while/lstm_cell_4/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_1894766_0while_lstm_cell_4_1894768_0while_lstm_cell_4_1894770_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894683?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:??????????
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????x

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_4_1894766while_lstm_cell_4_1894766_0"8
while_lstm_cell_4_1894768while_lstm_cell_4_1894768_0"8
while_lstm_cell_4_1894770while_lstm_cell_4_1894770_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?%
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896560

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????	s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????	?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????	: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895605

inputs+
batch_normalization_4_1895578:+
batch_normalization_4_1895580:+
batch_normalization_4_1895582:+
batch_normalization_4_1895584: 
lstm_4_1895587:x
lstm_4_1895589:x 
lstm_4_1895591:x!
dense_8_1895594:
dense_8_1895596:!
dense_9_1895599:
dense_9_1895601:
identity??-batch_normalization_4/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_1895578batch_normalization_4_1895580batch_normalization_4_1895582batch_normalization_4_1895584*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1895535?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0lstm_4_1895587lstm_4_1895589lstm_4_1895591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895477?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_1895594dense_8_1895596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1895103?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1895599dense_9_1895601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1895119w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1896400

inputsK
=batch_normalization_4_assignmovingavg_readvariableop_resource:M
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:I
;batch_normalization_4_batchnorm_mul_readvariableop_resource:E
7batch_normalization_4_batchnorm_readvariableop_resource:B
0lstm_4_lstm_cell_4_split_readvariableop_resource:x@
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:x<
*lstm_4_lstm_cell_4_readvariableop_resource:x8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity??%batch_normalization_4/AssignMovingAvg?4batch_normalization_4/AssignMovingAvg/ReadVariableOp?'batch_normalization_4/AssignMovingAvg_1?6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_4/batchnorm/ReadVariableOp?2batch_normalization_4/batchnorm/mul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?!lstm_4/lstm_cell_4/ReadVariableOp?#lstm_4/lstm_cell_4/ReadVariableOp_1?#lstm_4/lstm_cell_4/ReadVariableOp_2?#lstm_4/lstm_cell_4/ReadVariableOp_3?'lstm_4/lstm_cell_4/split/ReadVariableOp?)lstm_4/lstm_cell_4/split_1/ReadVariableOp?lstm_4/while?
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
"batch_normalization_4/moments/meanMeaninputs=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*"
_output_shapes
:?
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceinputs3batch_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????	?
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_4/batchnorm/mul_1Mulinputs'batch_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	?
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	e
lstm_4/ShapeShape)batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Y
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????j
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_4/transpose	Transpose)batch_normalization_4/batchnorm/add_1:z:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:	?????????R
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskg
"lstm_4/lstm_cell_4/ones_like/ShapeShapelstm_4/zeros:output:0*
T0*
_output_shapes
:g
"lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_4/lstm_cell_4/ones_likeFill+lstm_4/lstm_cell_4/ones_like/Shape:output:0+lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????e
 lstm_4/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_4/lstm_cell_4/dropout/MulMul%lstm_4/lstm_cell_4/ones_like:output:0)lstm_4/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????u
 lstm_4/lstm_cell_4/dropout/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
7lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform)lstm_4/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???n
)lstm_4/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
'lstm_4/lstm_cell_4/dropout/GreaterEqualGreaterEqual@lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniform:output:02lstm_4/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/dropout/CastCast+lstm_4/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
 lstm_4/lstm_cell_4/dropout/Mul_1Mul"lstm_4/lstm_cell_4/dropout/Mul:z:0#lstm_4/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????g
"lstm_4/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
 lstm_4/lstm_cell_4/dropout_1/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????w
"lstm_4/lstm_cell_4/dropout_1/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2?̷p
+lstm_4/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
)lstm_4/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_4/lstm_cell_4/dropout_1/CastCast-lstm_4/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_4/lstm_cell_4/dropout_1/Mul_1Mul$lstm_4/lstm_cell_4/dropout_1/Mul:z:0%lstm_4/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????g
"lstm_4/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
 lstm_4/lstm_cell_4/dropout_2/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????w
"lstm_4/lstm_cell_4/dropout_2/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???p
+lstm_4/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
)lstm_4/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_4/lstm_cell_4/dropout_2/CastCast-lstm_4/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_4/lstm_cell_4/dropout_2/Mul_1Mul$lstm_4/lstm_cell_4/dropout_2/Mul:z:0%lstm_4/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????g
"lstm_4/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
 lstm_4/lstm_cell_4/dropout_3/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????w
"lstm_4/lstm_cell_4/dropout_3/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???p
+lstm_4/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
)lstm_4/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_4/lstm_cell_4/dropout_3/CastCast-lstm_4/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_4/lstm_cell_4/dropout_3/Mul_1Mul$lstm_4/lstm_cell_4/dropout_3/Mul:z:0%lstm_4/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????d
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????f
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0$lstm_4/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:??????????
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0w
&lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_4/lstm_cell_4/strided_sliceStridedSlice)lstm_4/lstm_cell_4/ReadVariableOp:value:0/lstm_4/lstm_cell_4/strided_slice/stack:output:01lstm_4/lstm_cell_4/strided_slice/stack_1:output:01lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????s
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   {
*lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_4/lstm_cell_4/strided_slice_1StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_1:value:01lstm_4/lstm_cell_4/strided_slice_1/stack:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????w
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:??????????
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   {
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   {
*lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_4/lstm_cell_4/strided_slice_2StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_2:value:01lstm_4/lstm_cell_4/strided_slice_2/stack:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????o
lstm_4/lstm_cell_4/TanhTanhlstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0lstm_4/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   {
*lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_4/lstm_cell_4/strided_slice_3StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_3:value:01lstm_4/lstm_cell_4/strided_slice_3/stack:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????w
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????q
lstm_4/lstm_cell_4/Tanh_1Tanhlstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0lstm_4/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????u
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_4_lstm_cell_4_split_readvariableop_resource2lstm_4_lstm_cell_4_split_1_readvariableop_resource*lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_4_while_body_1896228*%
condR
lstm_4_while_cond_1896227*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0o
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	b
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_8/MatMulMatMullstm_4/strided_slice_3:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2F
!lstm_4/lstm_cell_4/ReadVariableOp!lstm_4/lstm_cell_4/ReadVariableOp2J
#lstm_4/lstm_cell_4/ReadVariableOp_1#lstm_4/lstm_cell_4/ReadVariableOp_12J
#lstm_4/lstm_cell_4/ReadVariableOp_2#lstm_4/lstm_cell_4/ReadVariableOp_22J
#lstm_4/lstm_cell_4/ReadVariableOp_3#lstm_4/lstm_cell_4/ReadVariableOp_32R
'lstm_4/lstm_cell_4/split/ReadVariableOp'lstm_4/lstm_cell_4/split/ReadVariableOp2V
)lstm_4/lstm_cell_4/split_1/ReadVariableOp)lstm_4/lstm_cell_4/split_1/ReadVariableOp2
lstm_4/whilelstm_4/while:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
while_cond_1895317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1895317___redundant_placeholder05
1while_while_cond_1895317___redundant_placeholder15
1while_while_cond_1895317___redundant_placeholder25
1while_while_cond_1895317___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
D__inference_dense_9_layer_call_and_return_conditional_losses_1897687

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?l
?	
while_body_1894957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_4_split_readvariableop_resource_0:xA
3while_lstm_cell_4_split_1_readvariableop_resource_0:x=
+while_lstm_cell_4_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_4_split_readvariableop_resource:x?
1while_lstm_cell_4_split_1_readvariableop_resource:x;
)while_lstm_cell_4_readvariableop_resource:x?? while/lstm_cell_4/ReadVariableOp?"while/lstm_cell_4/ReadVariableOp_1?"while/lstm_cell_4/ReadVariableOp_2?"while/lstm_cell_4/ReadVariableOp_3?&while/lstm_cell_4/split/ReadVariableOp?(while/lstm_cell_4/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
7__inference_batch_normalization_4_layer_call_fn_1896452

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1895535s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896472

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895126

inputs+
batch_normalization_4_1894847:+
batch_normalization_4_1894849:+
batch_normalization_4_1894851:+
batch_normalization_4_1894853: 
lstm_4_1895085:x
lstm_4_1895087:x 
lstm_4_1895089:x!
dense_8_1895104:
dense_8_1895106:!
dense_9_1895120:
dense_9_1895122:
identity??-batch_normalization_4/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_1894847batch_normalization_4_1894849batch_normalization_4_1894851batch_normalization_4_1894853*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894846?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0lstm_4_1895085lstm_4_1895087lstm_4_1895089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895084?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_1895104dense_8_1895106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1895103?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1895120dense_9_1895122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1895119w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
(__inference_lstm_4_layer_call_fn_1896593

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?z
?
lstm_4_while_body_1895924*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0J
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:xH
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:xD
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:x
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorH
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:xF
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:xB
0lstm_4_while_lstm_cell_4_readvariableop_resource:x??'lstm_4/while/lstm_cell_4/ReadVariableOp?)lstm_4/while/lstm_cell_4/ReadVariableOp_1?)lstm_4/while/lstm_cell_4/ReadVariableOp_2?)lstm_4/while/lstm_cell_4/ReadVariableOp_3?-lstm_4/while/lstm_cell_4/split/ReadVariableOp?/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp?
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
(lstm_4/while/lstm_cell_4/ones_like/ShapeShapelstm_4_while_placeholder_2*
T0*
_output_shapes
:m
(lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm_4/while/lstm_cell_4/ones_likeFill1lstm_4/while/lstm_cell_4/ones_like/Shape:output:01lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????j
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????l
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0}
,lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_4/while/lstm_cell_4/strided_sliceStridedSlice/lstm_4/while/lstm_cell_4/ReadVariableOp:value:05lstm_4/while/lstm_cell_4/strided_slice/stack:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   ?
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:07lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:??????????
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   ?
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   ?
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:07lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????{
lstm_4/while/lstm_cell_4/TanhTanh"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0!lstm_4/while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   ?
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:07lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????}
lstm_4/while/lstm_cell_4/Tanh_1Tanh"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0#lstm_4/while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ?
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ?
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ?
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_6:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_4/while/NoOpNoOp(^lstm_4/while/lstm_cell_4/ReadVariableOp*^lstm_4/while/lstm_cell_4/ReadVariableOp_1*^lstm_4/while/lstm_cell_4/ReadVariableOp_2*^lstm_4/while/lstm_cell_4/ReadVariableOp_3.^lstm_4/while/lstm_cell_4/split/ReadVariableOp0^lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"f
0lstm_4_while_lstm_cell_4_readvariableop_resource2lstm_4_while_lstm_cell_4_readvariableop_resource_0"v
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"r
6lstm_4_while_lstm_cell_4_split_readvariableop_resource8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"?
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2R
'lstm_4/while/lstm_cell_4/ReadVariableOp'lstm_4/while/lstm_cell_4/ReadVariableOp2V
)lstm_4/while/lstm_cell_4/ReadVariableOp_1)lstm_4/while/lstm_cell_4/ReadVariableOp_12V
)lstm_4/while/lstm_cell_4/ReadVariableOp_2)lstm_4/while/lstm_cell_4/ReadVariableOp_22V
)lstm_4/while/lstm_cell_4/ReadVariableOp_3)lstm_4/while/lstm_cell_4/ReadVariableOp_32^
-lstm_4/while/lstm_cell_4/split/ReadVariableOp-lstm_4/while/lstm_cell_4/split/ReadVariableOp2b
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?8
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1894811

inputs%
lstm_cell_4_1894729:x!
lstm_cell_4_1894731:x%
lstm_cell_4_1894733:x
identity??#lstm_cell_4/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_1894729lstm_cell_4_1894731lstm_cell_4_1894733*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894683n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_1894729lstm_cell_4_1894731lstm_cell_4_1894733*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1894742*
condR
while_cond_1894741*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????t
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_8_layer_call_fn_1897657

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1895103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_8_layer_call_and_return_conditional_losses_1895103

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?y
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1896833
inputs_0;
)lstm_cell_4_split_readvariableop_resource:x9
+lstm_cell_4_split_1_readvariableop_resource:x5
#lstm_cell_4_readvariableop_resource:x
identity??lstm_cell_4/ReadVariableOp?lstm_cell_4/ReadVariableOp_1?lstm_cell_4/ReadVariableOp_2?lstm_cell_4/ReadVariableOp_3? lstm_cell_4/split/ReadVariableOp?"lstm_cell_4/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:?????????x
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????~
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????}
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1896706*
condR
while_cond_1896705*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?P
?
 __inference__traced_save_1898040
file_prefix:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::: : : : : :x:x:x: : : : :::::::x:x:x:::::::x:x:x: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	
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

:x:$ 

_output_shapes

:x: 

_output_shapes
:x:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x: 

_output_shapes
:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:x:$% 

_output_shapes

:x: &

_output_shapes
:x:'

_output_shapes
: 
?%
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1895535

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????	s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????	h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????	f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????	?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????	: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
.__inference_sequential_4_layer_call_fn_1895151
input_5
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895126o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?	
?
lstm_4_while_cond_1896227*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1C
?lstm_4_while_lstm_4_while_cond_1896227___redundant_placeholder0C
?lstm_4_while_lstm_4_while_cond_1896227___redundant_placeholder1C
?lstm_4_while_lstm_4_while_cond_1896227___redundant_placeholder2C
?lstm_4_while_lstm_4_while_cond_1896227___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
(__inference_lstm_4_layer_call_fn_1896582
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1894811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?y
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897355

inputs;
)lstm_cell_4_split_readvariableop_resource:x9
+lstm_cell_4_split_1_readvariableop_resource:x5
#lstm_cell_4_readvariableop_resource:x
identity??lstm_cell_4/ReadVariableOp?lstm_cell_4/ReadVariableOp_1?lstm_cell_4/ReadVariableOp_2?lstm_cell_4/ReadVariableOp_3? lstm_cell_4/split/ReadVariableOp?"lstm_cell_4/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:?????????x
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:?????????~
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????}
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1897228*
condR
while_cond_1897227*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_4_layer_call_fn_1896439

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894846s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?8
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1894545

inputs%
lstm_cell_4_1894463:x!
lstm_cell_4_1894465:x%
lstm_cell_4_1894467:x
identity??#lstm_cell_4/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_1894463lstm_cell_4_1894465lstm_cell_4_1894467*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894462n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_1894463lstm_cell_4_1894465lstm_cell_4_1894467*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1894476*
condR
while_cond_1894475*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????t
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
ҥ
?
lstm_4_while_body_1896228*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0J
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:xH
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:xD
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:x
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorH
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:xF
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:xB
0lstm_4_while_lstm_cell_4_readvariableop_resource:x??'lstm_4/while/lstm_cell_4/ReadVariableOp?)lstm_4/while/lstm_cell_4/ReadVariableOp_1?)lstm_4/while/lstm_cell_4/ReadVariableOp_2?)lstm_4/while/lstm_cell_4/ReadVariableOp_3?-lstm_4/while/lstm_cell_4/split/ReadVariableOp?/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp?
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
(lstm_4/while/lstm_cell_4/ones_like/ShapeShapelstm_4_while_placeholder_2*
T0*
_output_shapes
:m
(lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm_4/while/lstm_cell_4/ones_likeFill1lstm_4/while/lstm_cell_4/ones_like/Shape:output:01lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????k
&lstm_4/while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
$lstm_4/while/lstm_cell_4/dropout/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:0/lstm_4/while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:??????????
&lstm_4/while/lstm_cell_4/dropout/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
=lstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform/lstm_4/while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???t
/lstm_4/while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
-lstm_4/while/lstm_cell_4/dropout/GreaterEqualGreaterEqualFlstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:08lstm_4/while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
%lstm_4/while/lstm_cell_4/dropout/CastCast1lstm_4/while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
&lstm_4/while/lstm_cell_4/dropout/Mul_1Mul(lstm_4/while/lstm_cell_4/dropout/Mul:z:0)lstm_4/while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????m
(lstm_4/while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
&lstm_4/while/lstm_cell_4/dropout_1/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_4/while/lstm_cell_4/dropout_1/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??sv
1lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
/lstm_4/while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_4/while/lstm_cell_4/dropout_1/CastCast3lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_4/while/lstm_cell_4/dropout_1/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_1/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????m
(lstm_4/while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
&lstm_4/while/lstm_cell_4/dropout_2/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_4/while/lstm_cell_4/dropout_2/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???v
1lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
/lstm_4/while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_4/while/lstm_cell_4/dropout_2/CastCast3lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_4/while/lstm_cell_4/dropout_2/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_2/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????m
(lstm_4/while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
&lstm_4/while/lstm_cell_4/dropout_3/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_4/while/lstm_cell_4/dropout_3/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2?Ŋv
1lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
/lstm_4/while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_4/while/lstm_cell_4/dropout_3/CastCast3lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_4/while/lstm_cell_4/dropout_3/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_3/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????j
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????l
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2*lstm_4/while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:??????????
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0}
,lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_4/while/lstm_cell_4/strided_sliceStridedSlice/lstm_4/while/lstm_cell_4/ReadVariableOp:value:05lstm_4/while/lstm_cell_4/strided_slice/stack:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   ?
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:07lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:??????????
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   ?
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   ?
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:07lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????{
lstm_4/while/lstm_cell_4/TanhTanh"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0!lstm_4/while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   ?
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:07lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????}
lstm_4/while/lstm_cell_4/Tanh_1Tanh"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0#lstm_4/while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ?
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ?
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ?
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_6:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_4/while/NoOpNoOp(^lstm_4/while/lstm_cell_4/ReadVariableOp*^lstm_4/while/lstm_cell_4/ReadVariableOp_1*^lstm_4/while/lstm_cell_4/ReadVariableOp_2*^lstm_4/while/lstm_cell_4/ReadVariableOp_3.^lstm_4/while/lstm_cell_4/split/ReadVariableOp0^lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"f
0lstm_4_while_lstm_cell_4_readvariableop_resource2lstm_4_while_lstm_cell_4_readvariableop_resource_0"v
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"r
6lstm_4_while_lstm_cell_4_split_readvariableop_resource8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"?
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2R
'lstm_4/while/lstm_cell_4/ReadVariableOp'lstm_4/while/lstm_cell_4/ReadVariableOp2V
)lstm_4/while/lstm_cell_4/ReadVariableOp_1)lstm_4/while/lstm_cell_4/ReadVariableOp_12V
)lstm_4/while/lstm_cell_4/ReadVariableOp_2)lstm_4/while/lstm_cell_4/ReadVariableOp_22V
)lstm_4/while/lstm_cell_4/ReadVariableOp_3)lstm_4/while/lstm_cell_4/ReadVariableOp_32^
-lstm_4/while/lstm_cell_4/split/ReadVariableOp-lstm_4/while/lstm_cell_4/split/ReadVariableOp2b
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_lstm_cell_4_layer_call_fn_1897704

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894462o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
7__inference_batch_normalization_4_layer_call_fn_1896413

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1894294|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?\
?
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894683

inputs

states
states_1/
split_readvariableop_resource:x-
split_1_readvariableop_resource:x)
readvariableop_resource:x
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:x*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????W
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????[
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????[
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????[
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Y
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?"
?
while_body_1894476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_1894500_0:x)
while_lstm_cell_4_1894502_0:x-
while_lstm_cell_4_1894504_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_1894500:x'
while_lstm_cell_4_1894502:x+
while_lstm_cell_4_1894504:x??)while/lstm_cell_4/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_1894500_0while_lstm_cell_4_1894502_0while_lstm_cell_4_1894504_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894462?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:??????????
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????x

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_4_1894500while_lstm_cell_4_1894500_0"8
while_lstm_cell_4_1894502while_lstm_cell_4_1894502_0"8
while_lstm_cell_4_1894504while_lstm_cell_4_1894504_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ؓ
?	
while_body_1896967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_4_split_readvariableop_resource_0:xA
3while_lstm_cell_4_split_1_readvariableop_resource_0:x=
+while_lstm_cell_4_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_4_split_readvariableop_resource:x?
1while_lstm_cell_4_split_1_readvariableop_resource:x;
)while_lstm_cell_4_readvariableop_resource:x?? while/lstm_cell_4/ReadVariableOp?"while/lstm_cell_4/ReadVariableOp_1?"while/lstm_cell_4/ReadVariableOp_2?"while/lstm_cell_4/ReadVariableOp_3?&while/lstm_cell_4/split/ReadVariableOp?(while/lstm_cell_4/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????d
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????s
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???m
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2Ķ?o
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??o
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???o
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_lstm_4_layer_call_fn_1896571
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1894545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
.__inference_sequential_4_layer_call_fn_1895779

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895126o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?%
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896506

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_1896966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1896966___redundant_placeholder05
1while_while_cond_1896966___redundant_placeholder15
1while_while_cond_1896966___redundant_placeholder25
1while_while_cond_1896966___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895477

inputs;
)lstm_cell_4_split_readvariableop_resource:x9
+lstm_cell_4_split_1_readvariableop_resource:x5
#lstm_cell_4_readvariableop_resource:x
identity??lstm_cell_4/ReadVariableOp?lstm_cell_4/ReadVariableOp_1?lstm_cell_4/ReadVariableOp_2?lstm_cell_4/ReadVariableOp_3? lstm_cell_4/split/ReadVariableOp?"lstm_cell_4/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????^
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????g
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???g
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???i
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???i
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???i
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????}
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1895318*
condR
while_cond_1895317*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:	?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????	[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
(__inference_lstm_4_layer_call_fn_1896604

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?>
?
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1894462

inputs

states
states_1/
split_readvariableop_resource:x-
split_1_readvariableop_resource:x)
readvariableop_resource:x
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:x*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????X
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:?????????Z
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:?????????Z
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:?????????Z
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:x*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Y
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
֓
?	
while_body_1897489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_4_split_readvariableop_resource_0:xA
3while_lstm_cell_4_split_1_readvariableop_resource_0:x=
+while_lstm_cell_4_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_4_split_readvariableop_resource:x?
1while_lstm_cell_4_split_1_readvariableop_resource:x;
)while_lstm_cell_4_readvariableop_resource:x?? while/lstm_cell_4/ReadVariableOp?"while/lstm_cell_4/ReadVariableOp_1?"while/lstm_cell_4/ReadVariableOp_2?"while/lstm_cell_4/ReadVariableOp_3?&while/lstm_cell_4/split/ReadVariableOp?(while/lstm_cell_4/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????d
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????s
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???m
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??)o
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??5o
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????u
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2??o
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?l
?	
while_body_1897228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_4_split_readvariableop_resource_0:xA
3while_lstm_cell_4_split_1_readvariableop_resource_0:x=
+while_lstm_cell_4_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_4_split_readvariableop_resource:x?
1while_lstm_cell_4_split_1_readvariableop_resource:x;
)while_lstm_cell_4_readvariableop_resource:x?? while/lstm_cell_4/ReadVariableOp?"while/lstm_cell_4/ReadVariableOp_1?"while/lstm_cell_4/ReadVariableOp_2?"while/lstm_cell_4/ReadVariableOp_3?&while/lstm_cell_4/split/ReadVariableOp?(while/lstm_cell_4/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes

:x*
dtype0?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes
:x*
dtype0?
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes

:x*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_8_layer_call_and_return_conditional_losses_1897668

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897126
inputs_0;
)lstm_cell_4_split_readvariableop_resource:x9
+lstm_cell_4_split_1_readvariableop_resource:x5
#lstm_cell_4_readvariableop_resource:x
identity??lstm_cell_4/ReadVariableOp?lstm_cell_4/ReadVariableOp_1?lstm_cell_4/ReadVariableOp_2?lstm_cell_4/ReadVariableOp_3? lstm_cell_4/split/ReadVariableOp?"lstm_cell_4/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????^
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????g
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2?ɡg
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2崂i
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???i
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed*
seed2???i
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes

:x*
dtype0?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes
:x*
dtype0?
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes

:x*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    Z   t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:?????????}
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1896967*
condR
while_cond_1896966*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895717
input_5+
batch_normalization_4_1895690:+
batch_normalization_4_1895692:+
batch_normalization_4_1895694:+
batch_normalization_4_1895696: 
lstm_4_1895699:x
lstm_4_1895701:x 
lstm_4_1895703:x!
dense_8_1895706:
dense_8_1895708:!
dense_9_1895711:
dense_9_1895713:
identity??-batch_normalization_4/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinput_5batch_normalization_4_1895690batch_normalization_4_1895692batch_normalization_4_1895694batch_normalization_4_1895696*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1895535?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0lstm_4_1895699lstm_4_1895701lstm_4_1895703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_4_layer_call_and_return_conditional_losses_1895477?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_1895706dense_8_1895708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1895103?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1895711dense_9_1895713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1895119w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_4/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????	: : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?
?
&sequential_4_lstm_4_while_cond_1894129D
@sequential_4_lstm_4_while_sequential_4_lstm_4_while_loop_counterJ
Fsequential_4_lstm_4_while_sequential_4_lstm_4_while_maximum_iterations)
%sequential_4_lstm_4_while_placeholder+
'sequential_4_lstm_4_while_placeholder_1+
'sequential_4_lstm_4_while_placeholder_2+
'sequential_4_lstm_4_while_placeholder_3F
Bsequential_4_lstm_4_while_less_sequential_4_lstm_4_strided_slice_1]
Ysequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_1894129___redundant_placeholder0]
Ysequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_1894129___redundant_placeholder1]
Ysequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_1894129___redundant_placeholder2]
Ysequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_1894129___redundant_placeholder3&
"sequential_4_lstm_4_while_identity
?
sequential_4/lstm_4/while/LessLess%sequential_4_lstm_4_while_placeholderBsequential_4_lstm_4_while_less_sequential_4_lstm_4_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/lstm_4/while/IdentityIdentity"sequential_4/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_lstm_4_while_identity+sequential_4/lstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_54
serving_default_input_5:0?????????	;
dense_90
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
o__call__
*p&call_and_return_all_conditional_losses
q_default_save_signature"
_tf_keras_sequential
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&iter

'beta_1

(beta_2
	)decay
*learning_ratem]m^m_m` ma!mb+mc,md-mevfvgvhvi vj!vk+vl,vm-vn"
	optimizer
n
0
1
2
3
+4
,5
-6
7
8
 9
!10"
trackable_list_wrapper
_
0
1
+2
,3
-4
5
6
 7
!8"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
o__call__
q_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
?
8
state_size

+kernel
,recurrent_kernel
-bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 :2dense_8/kernel
:2dense_8/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 :2dense_9/kernel
:2dense_9/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
"	variables
#trainable_variables
$regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)x2lstm_4/lstm_cell_4/kernel
5:3x2#lstm_4/lstm_cell_4/recurrent_kernel
%:#x2lstm_4/lstm_cell_4/bias
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
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
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
9	variables
:trainable_variables
;regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ttotal
	Ucount
V	variables
W	keras_api"
_tf_keras_metric
^
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
T0
U1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
.:,2"Adam/batch_normalization_4/gamma/m
-:+2!Adam/batch_normalization_4/beta/m
%:#2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
%:#2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
0:.x2 Adam/lstm_4/lstm_cell_4/kernel/m
::8x2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
*:(x2Adam/lstm_4/lstm_cell_4/bias/m
.:,2"Adam/batch_normalization_4/gamma/v
-:+2!Adam/batch_normalization_4/beta/v
%:#2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
%:#2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
0:.x2 Adam/lstm_4/lstm_cell_4/kernel/v
::8x2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
*:(x2Adam/lstm_4/lstm_cell_4/bias/v
?2?
.__inference_sequential_4_layer_call_fn_1895151
.__inference_sequential_4_layer_call_fn_1895779
.__inference_sequential_4_layer_call_fn_1895806
.__inference_sequential_4_layer_call_fn_1895657?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1896064
I__inference_sequential_4_layer_call_and_return_conditional_losses_1896400
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895687
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895717?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1894270input_5"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_4_layer_call_fn_1896413
7__inference_batch_normalization_4_layer_call_fn_1896426
7__inference_batch_normalization_4_layer_call_fn_1896439
7__inference_batch_normalization_4_layer_call_fn_1896452?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896472
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896506
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896526
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896560?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_lstm_4_layer_call_fn_1896571
(__inference_lstm_4_layer_call_fn_1896582
(__inference_lstm_4_layer_call_fn_1896593
(__inference_lstm_4_layer_call_fn_1896604?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1896833
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897126
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897355
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897648?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_8_layer_call_fn_1897657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_8_layer_call_and_return_conditional_losses_1897668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_9_layer_call_fn_1897677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_9_layer_call_and_return_conditional_losses_1897687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1895752input_5"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_lstm_cell_4_layer_call_fn_1897704
-__inference_lstm_cell_4_layer_call_fn_1897721?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1897796
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1897903?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
"__inference__wrapped_model_1894270v+-, !4?1
*?'
%?"
input_5?????????	
? "1?.
,
dense_9!?
dense_9??????????
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896472|@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896506|@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896526j7?4
-?*
$?!
inputs?????????	
p 
? ")?&
?
0?????????	
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1896560j7?4
-?*
$?!
inputs?????????	
p
? ")?&
?
0?????????	
? ?
7__inference_batch_normalization_4_layer_call_fn_1896413o@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
7__inference_batch_normalization_4_layer_call_fn_1896426o@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
7__inference_batch_normalization_4_layer_call_fn_1896439]7?4
-?*
$?!
inputs?????????	
p 
? "??????????	?
7__inference_batch_normalization_4_layer_call_fn_1896452]7?4
-?*
$?!
inputs?????????	
p
? "??????????	?
D__inference_dense_8_layer_call_and_return_conditional_losses_1897668\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_8_layer_call_fn_1897657O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_9_layer_call_and_return_conditional_losses_1897687\ !/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_9_layer_call_fn_1897677O !/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_lstm_4_layer_call_and_return_conditional_losses_1896833}+-,O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????
? ?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897126}+-,O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????
? ?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897355m+-,??<
5?2
$?!
inputs?????????	

 
p 

 
? "%?"
?
0?????????
? ?
C__inference_lstm_4_layer_call_and_return_conditional_losses_1897648m+-,??<
5?2
$?!
inputs?????????	

 
p

 
? "%?"
?
0?????????
? ?
(__inference_lstm_4_layer_call_fn_1896571p+-,O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "???????????
(__inference_lstm_4_layer_call_fn_1896582p+-,O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "???????????
(__inference_lstm_4_layer_call_fn_1896593`+-,??<
5?2
$?!
inputs?????????	

 
p 

 
? "???????????
(__inference_lstm_4_layer_call_fn_1896604`+-,??<
5?2
$?!
inputs?????????	

 
p

 
? "???????????
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1897796?+-,??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
H__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1897903?+-,??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
-__inference_lstm_cell_4_layer_call_fn_1897704?+-,??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
-__inference_lstm_cell_4_layer_call_fn_1897721?+-,??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895687r+-, !<?9
2?/
%?"
input_5?????????	
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1895717r+-, !<?9
2?/
%?"
input_5?????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1896064q+-, !;?8
1?.
$?!
inputs?????????	
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_1896400q+-, !;?8
1?.
$?!
inputs?????????	
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_4_layer_call_fn_1895151e+-, !<?9
2?/
%?"
input_5?????????	
p 

 
? "???????????
.__inference_sequential_4_layer_call_fn_1895657e+-, !<?9
2?/
%?"
input_5?????????	
p

 
? "???????????
.__inference_sequential_4_layer_call_fn_1895779d+-, !;?8
1?.
$?!
inputs?????????	
p 

 
? "???????????
.__inference_sequential_4_layer_call_fn_1895806d+-, !;?8
1?.
$?!
inputs?????????	
p

 
? "???????????
%__inference_signature_wrapper_1895752?+-, !??<
? 
5?2
0
input_5%?"
input_5?????????	"1?.
,
dense_9!?
dense_9?????????