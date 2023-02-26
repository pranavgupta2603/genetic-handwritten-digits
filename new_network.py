from math import floor
def calculate_convolution_output_shape(input_shape, filter_shape, stride, padding, pool, nfilters=1, last_conv=False):
    
    height = (input_shape[0] - filter_shape + (2 * padding)) // stride + 1
    width = (input_shape[1] - filter_shape + (2 * padding)) // stride + 1

    height = floor(height / pool)
    width = floor(width / pool)
    # Calculate the output shape of the convolutional layer
    output_shape = (height, width)
    if last_conv:
        output_flatten = height*width*nfilters
        return output_flatten
    else:
        return output_shape
#output = calculate_convolution_output_shape(input_shape=(28, 28), filter_shape=(4, 4), stride=1, padding=2, pool=2)
#output = calculate_convolution_output_shape(input_shape=output, filter_shape=(1, 1), stride=1, padding=2, pool=2, nfilters=48, last_conv=True)

#print(output)
    
