cmake_minimum_required(VERSION 3.10)
project(2ndGeneticAlg)

set(CMAKE_CXX_STANDARD 20)

# Добавляем все исходные файлы в проект
set(SOURCE_FILES
        main.cpp
)

add_executable(2ndGeneticAlg ${SOURCE_FILES})

# Убедитесь, что правильно указываете необходимые библиотеки
# target_link_libraries(2ndGeneticAlg <libraries>)
