# -*- coding: utf-8 -*-


import tensorflow as tf

tf.reset_default_graph() 

with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v", [1])

print("sp:",sp.name)    
print("var1:",var1.name)      

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])
          
        with tf.variable_scope("") :
            var4 = tf.get_variable("v4", [1])
            
print("sp1:",sp1.name)  
print("var2:",var2.name)

# var3 在scope2下 但是输出仍是scope1 没有改变
print("var3:",var3.name)
# var4 同var3 表名sp没有收到外界的限制
print("var4:",var4.name)

with tf.variable_scope("scope"):
    # name_scope只能限制op 不能限制变量的命名
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
        # tf.name_scope("") 使用空字符将作用域返回到顶层
        with tf.name_scope(""):
            y = 1.0 + v
print("v:",v.name)  
print("x.op:",x.op.name)
print("y.op:",y.op.name)
