random_transforms_limit_dict ={

"random_crop":
{
    "height":{"min_limit":0.05, "max_limit":0.5},
    "width":{"min_limit":0.05, "max_limit":0.5}
},

"random_scale":
{
    "min_limit":0.1,
    "max_limit":10
},

"random_flip":
{
    "limit":None,
},


"random_rotate": 
{
    "min_limit":-90, 
    "max_limit":90
},

"random_shift_scale_rotate":
{
    "angle": {"min_limit":-90, "max_limit":90},
    "scale": {"min_limit":0.1, "max_limit":10},
    "dx": {"min_limit":0, "max_limit":"image_width/2"},
    "dy": {"min_limit":0, "max_limit":"image_height/2"}, 
}

#"random_resize":
#{
#    "width":{"min_limit":0.1, "max_limit":10},
#   "height":{"min_limit":0.1, "max_limit":10},
#    "interpolation":{"data_type":["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA", "cv2.INTER_LANCZOS4"],"default":"cv2.INTER_LINEAR"}
#}

}