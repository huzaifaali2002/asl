word_list = ["0", "1", "2", "Hello", "World", "In", "We", "Leave", "No One", "Behind"]
word_dict = {}
for i in range(len(word_list)):
    word_dict[i] = word_list[i]

train_data_path = r'C:\Huzaifa\HR\asl\train'
test_data_path = r"C:\Huzaifa\HR\asl\test"

batch = len(word_list)

bg = None
weights_for_background = 0.5

hand_zone_top = 40
hand_zone_bottom = 450
hand_zone_right = 300
hand_zone_left = 600

frame_num = 0
images_taken = 0

#insert element being added to dataset 
element = str("Hello") 