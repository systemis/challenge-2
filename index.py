# from imutils.perspective import four_point_transform
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
from os import listdir 
from os.path  import isfile, join 

data_name_list = [f for f in listdir('data') if isfile(join('data', f))]

height = 600 
width = 463

# 463, 600

character_table = ['A', 'B', 'C', 'D', 'E']
answer = 5 
questions_part = 5
total_part = 6
ques_each_side = 30
total_question = 60

statistical_table = [0] * total_question

def generateColumns(): 
  columns = ['Student ID', 'Surname', 'First Name', 'Code']
  for i in range(60): columns.append('Question: '+str((i+1)))
  columns.append('Score')
  return columns

def grid(): 
  _grid = [] 
  for i in range(total_question): 
    _grid.append([])
    for j in range(answer): 
      _grid[i].append(0)
  return _grid

def show_images(images):
    for image in images: 
        imshow = plt.imshow(image)
        plt.show()

img_filenames = data_name_list
img_list = list()
student_data_list = list()
for name in img_filenames: 
  if name == '.DS_Store': continue
  img = plt.imread('data/' + name)
  img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
  img = img[90:, :]
  img_list.append(img)
  
  name = name.replace('.png', '').split('_')
  if len(name) < 4: 
    print(name)
    continue
  student = [name[0], name[1], name[2], name[3]]
  student_data_list.append(student)
  
# img_test = img_list[0]
# show_images([img_test, img_list[1]])

def get_rect_list(contours): 
  rect_list = list() 
  for p_c in contours: 
    arc_length = cv2.arcLength(p_c, True)
    approx = cv2.approxPolyDP(p_c, arc_length * 0.02, True)
    if approx == 4: 
      rect_list.append(p_c)
  return rect_list

def write_csv_file(): 
  if len(student_data_list) == 0: 
    return False
  students = pd.DataFrame(student_data_list, columns=['Student ID', 'Surname', 'First Name', 'Code'])
  students.to_csv('student.csv', index=False, sep=';')

def check_part(sides, side_pos, pc, result_data): 
  part = sides[side_pos][(80*pc):80*(pc+1), :]
  part = part[5:70, :]
  gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
  _, threshold = cv2.threshold(gray, 20, 5, cv2.THRESH_BINARY_INV)
  
  row_height = part.shape[0] // questions_part
  col_width = part.shape[1] // answer
  part_gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
  # show_images([part])
  part = part_gray
  for i in range(questions_part): 
    current_question = (questions_part*pc + i + 1) + ques_each_side*side_pos
    row = part[row_height*i:row_height*(i+1), :]
    for j in range(answer): 
      col = row[:, col_width*j:col_width*(j+1)]
      count = cv2.countNonZero(col)
      if count < 300: 
        result_data[current_question - 1][j] = 1
        # print('Check question  ', current_question, ' at: ', character_table[j])
        # print(result_data[current_question - 1])

# def grade(img): # Return score
def detect_checked(img, only_firstfive=False): 
  # img = img_list[0]
  img_copy = img.copy()
  left_side = img[:, 80:200]
  right_side = img[:, 290:410]
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

  left_side = left_side[:490, :]
  right_side = right_side[:490, :]
  sides = [left_side, right_side]

  result_data = grid()
  if only_firstfive: 
    for side_pos in range(1): 
      part_height = sides[side_pos].shape[0] // total_part
      for pc in range(1): 
        check_part(sides, side_pos, pc, result_data)
    return result_data
  for side_pos in range(2): 
    part_height = sides[side_pos].shape[0] // total_part
    for pc in range(total_part): 
      check_part(sides, side_pos, pc, result_data)
  return result_data

'''
  Detect answer data from image
  Use same algorithm with grading to get data
'''
def detectAnswerData(): 
  answer_data = ['' for i in range(total_question)]
  answer_image = plt.imread('answer/3A.png')
  answer_image = cv2.resize(answer_image, (width, height), interpolation = cv2.INTER_AREA)
  answer_image = answer_image[90:, :]
  checked_data = detect_checked(answer_image) 
  # show_images([answer_image])
  for i in range(len(checked_data)): 
    for j in range(len(checked_data[i])): 
      if checked_data[i][j]: 
        answer_data[i] = character_table[j]
  return answer_data

answer_data = detectAnswerData()

# print(answer_data)

'''
  - Grade based on comparison of checked_data 
  - one-hot array [0, 0, 0, 0, 1] - 1 represent for checked
'''
def grade(checked_data, student, is_statistical=False): 
  score = 0 
  check =0 
  for (question_index, question_data) in enumerate(checked_data): 
    for j in range(len(question_data)): 
      if question_data[j] == 1: 
        result = character_table[j]
        student.append(result)
        check += 1 
        if result == answer_data[question_index]: 
          if is_statistical: statistical_table[question_index] += 1 
          score += 1 
        break
  return score

def checkDataInFirstFive(): 
  student_data_copy = student_data_list.copy()
  for (i, image) in enumerate(img_list): 
    student = student_data_copy[i].copy()
    checked_data = detect_checked(image, only_firstfive=True)
    checked_data = checked_data[:5]
    for question_data in checked_data: 
      for j in range(len(question_data)): 
        if question_data[j] == 1: 
          student.append(character_table[j])
    student_data_copy[i] = student

  columns = ['Student ID', 'Surname', 'First Name', 'Code']
  for i in range(5): columns.append('Question: '+str((i+1)))
  data_frame = pd.DataFrame(student_data_copy, columns=columns)
  print('--- Checked of first five question of all student: -----')
  print(data_frame)
  print('\n')

def generateStudentScore(student_index=0): 
  student_data_copy = student_data_list.copy()
  score = 0
  i = student_index
  image = img_list[student_index]
  student = student_data_copy[student_index].copy()
  checked_data = detect_checked(image, only_firstfive=False)
  
  checked_data = checked_data


  score = grade(checked_data, student)
  score = (score / total_question) * 100
  student.append(score)
  print(len(student), len(checked_data))
  student_data_copy[i] = student
  
  student_data_copy = student_data_copy[student_index:student_index+1]
  columns = generateColumns()
  data_frame = pd.DataFrame(student_data_copy, columns=columns)
  print('---- Checked data of student: ', student[2], '----')
  print(data_frame)
  print('\n')

def gradingStudentInClass(): 
  global statistical_table
  student_data_copy = student_data_list.copy()
  for i in range(len(student_data_copy)): 
    image = img_list[i]
    student = student_data_copy[i].copy()
    checked_data = detect_checked(image)
    score = grade(checked_data, student, True)
    score = (score / total_question) * 100
    student.append(score)
    student_data_copy[i] = student
  
  columns = generateColumns()
  data_frame = pd.DataFrame(student_data_copy, columns=columns)

  grading_columns = ['ID', 'Score']
  grading_data = [[item[0], item[-1]] for item in student_data_copy]
  print('---- Generate grading score of class: Done ----\n')
  # print(student_data_copy)
  for i in range(len(student_data_copy)): 
    if len(student_data_copy[i]) > 65: 
      print(student_data_copy[i])
  grading_data_frame = pd.DataFrame(grading_data, columns=grading_columns)
  grading_data_frame.to_csv('grading.csv')
  return data_frame

# generateStudentScore(7)
# gradingStudentInClass()

def analysisScore(): 
  table = statistical_table.copy()
  for i in range(len(table)): 
    table[i] = {'q': i + 1, 'count': table[i]}
  for i in range(len(table)): 
    for j in range(i + 1, len(table)): 
      if  table[i]['count'] > table[j]['count']: 
        tmp = table[i]
        table[i] = table[j]
        table[j] = tmp
  return table
  
def findThreeMostDiffQuestion(): 
  analysis_score_table = analysisScore() 
  three_most_dff = [item['q'] for item in analysis_score_table[:3]]
  print('Three most difficult questions: ', ", ".join([str(item) for item in three_most_dff]))

def finalResult(data_frame): 
  pass_student_count = 0 
  for i in range(0, len(data_frame)): 
    student_score = data_frame.iloc[i]['Score']
    if student_score > 40.0: 
      pass_student_count += 1 
  
  print('Number of student having score pass: ', pass_student_count)
  print('Final result of class: ', 'Pass' if (pass_student_count/len(student_data_list))*100 > 50.0 else 'Failed')
  return pass_student_count

# print(answer_data)

'''Question 2: Generating student.csv'''
write_csv_file()

'''Question 3: Generating the first 5 answers of all student '''
checkDataInFirstFive() 

'''Question 4: Generating all answers of one student'''
generateStudentScore(20)

'''Question 5: Generating grading.csv'''
student_data_frame = gradingStudentInClass()

'''Question 6: Summary which 3 questions are the most difficult. '''
findThreeMostDiffQuestion()

'''Question 7: Generating the final result (pass/fail) of the class.'''
finalResult(student_data_frame)