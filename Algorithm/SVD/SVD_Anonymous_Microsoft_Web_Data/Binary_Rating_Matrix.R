library('data.table') # manipulates the data

file_in <- "../../../Datasets/anonymous-msweb.csv"

table_in <- read.csv(file_in, header = FALSE)

table_users <- table_in[, 1:2]

# Convert it into a data table

table_users <- data.table(table_users)

# assign the column names and select the rows containing either users or items

setnames(table_users, 1:2, c("category", "value"))
table_users <- table_users[category %in% c("C", "V")]

table_users[, chunk_user := cumsum(category == "C")]

table_long <- table_users[, list(user = value[1], item = value[-1]), by = "chunk_user"]

table_long[, value := 1]
table_wide <- reshape(data = table_long,   # This is the table in long format.
                      direction = "wide",  # This shows whenever we are reshaping from long to wide or otherwise.
                      idvar = "user",      # This is the variable identifying the group, which, in this case, is the user.
                      timevar = "item",    # This is the variable identifying the record within the same group. In this case, it's the item.
                      v.names = "value")   # This is name of the values. In this ase, it's the rating that is always equal to one. Missing user-item combinations will be NA values.

#  keep only the columns containing ratings. (the user name will be the matrix row names)
vector_users <- table_wide[, user]
table_wide[, user := NULL]
table_wide[, chunk_user := NULL]

# To have the column names equal to the item names
setnames(x = table_wide,
         old = names(table_wide),
         new = substring(names(table_wide), 7))

# Set the row names equal to the user names
matrix_wide <- as.matrix(table_wide)
rownames(matrix_wide) <- vector_users

# coercing matrix_wide into a binary rating matrix

matrix_wide[is.na(matrix_wide)] <- 0
ratings_matrix <- as(matrix_wide, "binaryRatingMatrix")

image(ratings_matrix[1:50, 1:50], main = "Binary rating matrix")

# -- save ratings matrix --

df = data.frame(matrix_wide)

#save(df, file = 'test.csv')
write.csv(df, file = '../../../Datasets/MS_ratings_matrix.csv')
#write.csv(df, file = 'test.csv', row.names = FALSE, col.names = FALSE, sep = ',')
#write.table(matrix_wide, file = 'test.csv')
