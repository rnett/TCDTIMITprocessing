import concurrent.futures
import os.path


def silentremove (filename):
    try:
        os.remove(filename)
    except OSError as e:  # name the Exception `e`
        pass #print("Failed with:", e.strerror)  # look what it says
    return 0
    
# create a list with paths of all zip files
def createZipList(rootDir):
    zipList= []
    for root, dirs, files in os.walk(rootDir):
        for fname in files:
            if ".zip" in fname:
                path = ''.join([root, os.sep, fname])
                #print('Found zip: %s' % path)
                zipList.append(path)
    return zipList

# define func to unzip one file
import zipfile
def extractZip(filePath, delete=False):
    targetDir = os.path.dirname(filePath)
    with zipfile.ZipFile(filePath, "r") as z:
        z.extractall(targetDir)
    if (delete):
        silentremove(filePath)
    return 0
    
######################################
########## Main Function #############
######################################

rootDir = "/home/rnett/TCDTIMIT/data/volunteers"
batchSize = 1 # 1 or 2 recommended

zipList= createZipList(rootDir)
print("\n".join(zipList))

deleteZips = True
batchIndex = 0
executor = concurrent.futures.ProcessPoolExecutor(batchSize)
running = 1

while running:
    # get commands for current batch
    if batchIndex + batchSize > len(zipList):
        print("Processing LAST BATCH...")
        running = 0
        currentZips = zipList[batchIndex:]  # till the end
    else:
        currentZips = zipList[batchIndex:batchIndex + batchSize]

    # execute the commands
    futures = []
    for i in range(len(currentZips)):
        filePath = currentZips[i]
        print("Unzipping ", filePath)
        futures.append(executor.submit(extractZip, filePath, deleteZips))
    concurrent.futures.wait(futures)

    # update the batchIndex
    batchIndex += batchSize

    print("One batch complete.")
    print("---------------------------------")

print("All done!")
